import underworld as uw
from underworld import function as fn
import math

from planetengine.utilities import Grouper

def build(
        res = 64,
        f = 0.54,
        aspect = 1.,
        length = 1.,
        Ra = 1e7,
        heating = 1.,
        surfT = 0.,
        deltaT = 1.,
        diffusivity = 1.,
        buoyancy = 1.,
        creep = 1.,
        creep_sR = 3e4,
        tau = 1e5,
        tau_bR = 100.,
        cont_buoyancy_mR = 2.,
        cont_creep_mR = 2.,
        cont_creep_sR_mR = 2.,
        cont_tau_mR = 2.,
        cont_tau_bR_mR = 2.,
        cont_heating_mR = 2.,
        cont_diffusivity_mR = 0.5,
        periodic = False,
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

    swarm = uw.swarm.Swarm(mesh = mesh, particleEscape = True)
    swarm.populate_using_layout(
        uw.swarm.layouts.PerCellSpaceFillerLayout(
            swarm, 12
            )
        )
    materialVar = swarm.add_variable(dataType = "int", count = 1)

    repopulator = uw.swarm.PopulationControl(
        swarm,
        aggressive = True,
        splitThreshold = 0.15,
        maxDeletions = 2,
        maxSplits = 10,
        particlesPerCell = 10
        )

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

    baseT = surfT + deltaT

    scaledTFn = (temperatureField - surfT) / deltaT

    buoyancyFn = scaledTFn * Ra * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: buoyancy,
            1: buoyancy * cont_buoyancy_mR,
            }
        )

    diffusivityFn = fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: diffusivity,
            1: diffusivity * cont_diffusivity_mR,
            }
        )

    heatingFn = fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: heating,
            1: heating * cont_heating_mR,
            }
        )

    ### RHEOLOGY ###

    vc = uw.mesh.MeshVariable(mesh = mesh, nodeDofCount = 2)
    vc_eqNum = uw.systems.sle.EqNumber(vc, False )
    vcVec = uw.systems.sle.SolutionVector(vc, vc_eqNum)

    depthFn = (mesh.radialLengths[1] - mesh.radiusFn) / length

    yieldStressFn = fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: tau * (1. + (tau_bR - 1) * depthFn),
            1: tau * cont_tau_mR * (1. + (tau_bR * cont_tau_bR_mR - 1) * depthFn)
            }
        )

    secInvFn = fn.tensor.second_invariant(
        fn.tensor.symmetric(
            vc.fn_gradient
            )
        )

    plasticViscFn = yieldStressFn / (2. * secInvFn + 1e-18)

    creepViscFn = fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: creep * fn.math.pow(
                fn.misc.constant(creep_sR),
                -1. * (temperatureField - baseT)
                ),
            1: creep * cont_creep_mR * fn.math.pow(
                fn.misc.constant(creep_sR * cont_creep_sR_mR),
                -1. * (temperatureField - baseT)
                ),
            }
        )

    viscosityFn = fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: fn.misc.max(
                creep,
                fn.misc.min(
                    creep * creep_sR,
                    fn.misc.min(
                        creepViscFn,
                        plasticViscFn,
                        )
                    )
                ),
            1: fn.misc.max(
                creep * cont_creep_mR,
                fn.misc.min(
                    creep * creep_sR * cont_creep_mR * cont_creep_sR_mR,
                    fn.misc.min(
                        creepViscFn,
                        plasticViscFn,
                        )
                    )
                ),
            }
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

    advector = uw.systems.SwarmAdvector(
        swarm = swarm,
        velocityField = vc,
        order = 2,
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
        dt = min(advDiff.get_max_dt(), advector.get_max_dt())
        advDiff.integrate(dt)
        advector.integrate(dt)
        repopulator.repopulate()
        modeltime.value += dt
        step.value += 1

    def iterate():
        integrate()
        solve()

    ### HOUSEKEEPING: IMPORTANT! ###

    varsOfState = {
        'temperatureField': temperatureField,
        'materialVar': materialVar
        }
    varScales = {'temperatureField': [[surfT, baseT]]}
    varBounds = {'temperatureField': [[surfT, baseT, '.', '.']]}

    return Grouper(locals())
