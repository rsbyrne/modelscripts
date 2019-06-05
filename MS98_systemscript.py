import underworld as uw
from underworld import function as fn
import math

from planetengine.utilities import Grouper

def build(
        res = 64,
        f = 0.54,
        aspect = 1.,
        periodic = False,
        heating = 0.,
        Ra = 1e7,
        surfEta = 3e4,
        tau0 = 4e5,
        tau1 = 1e7,
        ):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### MESH & MESH VARIABLES ###

    maxf = 0.99999
    if f == 'max':
        f = maxf
    else:
        assert f < maxf

    length = 1.
    outerRad = 1. / (1. - f)
    radii = (outerRad - length, outerRad)

    maxAspect = math.pi * sum(radii) / length
    if aspect == 'max':
        aspect = maxAspect
        periodic = True
    else:
        assert aspect < maxAspect
        periodic = False

    width = length**2 * aspect * 2. / (radii[1]**2 - radii[0]**2)
    midpoint = math.pi / 2.
    angExtentRaw = (midpoint - 0.5 * width, midpoint + 0.5 * width)
    angExtentDeg = [item * 180. / math.pi for item in angExtentRaw]
    angularExtent = [
        max(0., angExtentDeg[0]),
        min(360., angExtentDeg[1] + abs(min(0., angExtentDeg[0])))
        ]
    angLen = angExtentRaw[1] - angExtentRaw[0]

    assert res % 4 == 0
    radRes = res
    angRes = 4 * int(angLen * (int(radRes * radii[1] / length)) / 4)
    elementRes = (radRes, angRes)

    mesh = uw.mesh.FeMesh_Annulus(
        elementRes = elementRes,
        radialLengths = radii,
        angularExtent = angularExtent,
        periodic = [False, periodic]
        )

    temperatureField = uw.mesh.MeshVariable(mesh, 1)
    temperatureDotField = uw.mesh.MeshVariable(mesh, 1)
    pressureField = uw.mesh.MeshVariable(mesh.subMesh, 1)
    velocityField = uw.mesh.MeshVariable(mesh, 2)

    ### BOUNDARIES ###
    
    inner = mesh.specialSets["inner"]
    outer = mesh.specialSets["outer"]
    sides = mesh.specialSets["MaxJ_VertexSet"] + mesh.specialSets["MinJ_VertexSet"]

    if periodic:
        velBC = uw.conditions.RotatedDirichletCondition(
            variable = velocityField,
            indexSetsPerDof = (inner + outer, None),
            basis_vectors = (mesh.bnd_vec_normal, mesh.bnd_vec_tangent)
            )
    else:
        velBC = uw.conditions.RotatedDirichletCondition(
            variable = velocityField,
            indexSetsPerDof = (inner + outer, sides),
            basis_vectors = (mesh.bnd_vec_normal, mesh.bnd_vec_tangent)
            )

    tempBC = uw.conditions.DirichletCondition(
        variable = temperatureField,
        indexSetsPerDof = (inner + outer,)
        )

    ### FUNCTIONS ###

    buoyancyFn = Ra * temperatureField

    diffusivityFn = 1.

    heatingFn = heating

    ### RHEOLOGY ###

    creepViscFn = fn.math.pow(
        surfEta,
        1. - temperatureField
        )

    depthFn = mesh.radialLengths[1] - mesh.radiusFn
    yieldStressFn = tau0 + depthFn * tau1
    vc = uw.mesh.MeshVariable(mesh = mesh, nodeDofCount = 2)
    vc_eqNum = uw.systems.sle.EqNumber(vc, False )
    vcVec = uw.systems.sle.SolutionVector(vc, vc_eqNum)
    secInvFn = fn.tensor.second_invariant(
        fn.tensor.symmetric(
            vc.fn_gradient
            )
        )
    plasticViscFn = yieldStressFn / (2. * secInvFn + 1e-18)

    viscosityFn = fn.misc.min(
        creepViscFn,
        plasticViscFn
        ) + 0. * velocityField[0]

    ### SYSTEMS ###

    stokes = uw.systems.Stokes(
        velocityField = velocityField,
        pressureField = pressureField,
        conditions = [velBC,],
        fn_viscosity = viscosityFn,
        fn_bodyforce = buoyancyFn * mesh.unitvec_r_Fn,
        _removeBCs = False,
        )

    solver = uw.systems.Solver(stokes)

    advDiff = uw.systems.AdvectionDiffusion(
        phiField = temperatureField,
        phiDotField = temperatureDotField,
        velocityField = vc,
        fn_diffusivity = diffusivityFn,
        fn_sourceTerm = heatingFn,
        conditions = [tempBC,]
        )

    step = fn.misc.constant(0)
    modeltime = fn.misc.constant(0.)

    ### SOLVING ###

    def postSolve():
        # realign solution using the rotation matrix on stokes
        uw.libUnderworld.Underworld.AXequalsY(
            stokes._rot._cself,
            stokes._velocitySol._cself,
            vcVec._cself,
            False
            )
        # remove null space - the solid body rotation velocity contribution
        uw.libUnderworld.StgFEM.SolutionVector_RemoveVectorSpace(
            stokes._velocitySol._cself, 
            stokes._vnsVec._cself
            )

    def solve():
        velocityField.data[:] = 0.
        solver.solve(
            nonLinearIterate = True,
            callback_post_solve = postSolve,
            )
        uw.libUnderworld.Underworld.AXequalsX(
            stokes._rot._cself,
            stokes._velocitySol._cself,
            False
            )

    def integrate():
        dt = advDiff.get_max_dt()
        advDiff.integrate(dt)
        modeltime.value += dt
        step.value += 1

    def iterate():
        integrate()
        solve()

    ### HOUSEKEEPING: IMPORTANT! ###

    varsOfState = {'temperatureField': temperatureField}
    varScales = {'temperatureField': [[0., 1.]]}
    varBounds = {'temperatureField': [[0., 1., '.', '.']]}

    return Grouper(locals())
