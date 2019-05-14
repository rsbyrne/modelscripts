import underworld as uw
from underworld import function as fn
import math

from planetengine.utilities import Grouper

def build(
        res = 64,
        f = 0.54,
        aspect = 1.,
        length = 1.,
        periodic = False,
        dirichlet = True,
        heating = 0.,
        diffusivity = 1.,
        buoyancy = 1.,
        buoyancy_bR = 1e7,
        creep = 1.,
        creep_sR = 3e4,
        tau0 = 4e5,
        tau1 = 1e7,
        cont_heating_mR = 2.,
        cont_diffusivity_mR = 0.5,
        cont_buoyancy_mR = 2.,
        cont_buoyancy_bR_mR = 1.,
        cont_creep_mR = 2.,
        cont_creep_sR_mR = 1.,
        cont_tau0_mR = 2.,
        cont_tau1_mR = 1.,
        solidus_refT = 1.,
        solidus_zCoef = 0.,
        ):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### MESH & MESH VARIABLES ###

    f = max(0.00001, min(0.99999, f))
    inputs['f'] = f

    outerRad = 1. / (1. - f)
    radii = (outerRad - length, outerRad)

    maxAspect = math.pi * sum(radii) / length
    aspect = min(aspect, maxAspect)
    inputs['aspect'] = aspect
    if aspect == maxAspect:
        periodic = True
        inputs['periodic'] = periodic

    width = length**2 * aspect * 2. / (radii[1]**2 - radii[0]**2)
    midpoint = math.pi / 2.
    angExtentRaw = (midpoint - 0.5 * width, midpoint + 0.5 * width)
    angExtentDeg = [item * 180. / math.pi for item in angExtentRaw]
    angularExtent = [
        max(0., angExtentDeg[0]),
        min(360., angExtentDeg[1] + abs(min(0., angExtentDeg[0])))
        ]
    angLen = angExtentRaw[1] - angExtentRaw[0]

    radRes = max(16, int(res / 16) * 16)
    inputs['res'] = radRes
    angRes = 16 * int(angLen * (int(radRes * radii[1] / length)) / 16)
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

    if dirichlet:
        tempBC = uw.conditions.DirichletCondition(
            variable = temperatureField,
            indexSetsPerDof = (inner + outer,)
            )
    else:
        tempBC = uw.conditions.NeumannCondition(
            fn_flux = 0.,
            variable = temperatureField,
            indexSetsPerDof = (inner + outer + sides,)
            )

    ### FUNCTIONS ###

    depthFn = (mesh.radialLengths[1] - mesh.radiusFn) \
        / (mesh.radialLengths[1] - mesh.radialLengths[0])

    refBuoyancyFn = buoyancy * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_buoyancy_mR,
            }
        )

    thermalBuoyancyFn = temperatureField * buoyancy_bR * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_buoyancy_bR_mR,
            }
        )

    buoyancyFn = refBuoyancyFn * (1. + thermalBuoyancyFn)

    diffusivityFn = diffusivity * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_diffusivity_mR,
            }
        )

    heatingFn = heating * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_heating_mR,
            }
        )

    ### RHEOLOGY ###

    creepFn = creep * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_creep_mR
            }
        )

    creepSrFn = creep_sR * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_creep_sR_mR
            }
        )

    creepViscFn = creepFn / fn.math.pow(creepSrFn, temperatureField - 1.)

    cohesiveYieldFn = tau0 * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_tau0_mR,
            }
        )

    depthYieldFn = depthFn * tau1 * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: 1.,
            1: cont_tau1_mR,
            }
        )

    yieldStressFn = cohesiveYieldFn + depthYieldFn

    vc = uw.mesh.MeshVariable(mesh = mesh, nodeDofCount = 2)
    vc_eqNum = uw.systems.sle.EqNumber(vc, False )
    vcVec = uw.systems.sle.SolutionVector(vc, vc_eqNum)

    secInvFn = fn.tensor.second_invariant(
        fn.tensor.symmetric(
            vc.fn_gradient
            )
        )

    plasticViscFn = yieldStressFn / (2. * secInvFn + 1e-18)

    viscosityFn = fn.misc.min(creepViscFn, plasticViscFn) + (0. * velocityField[0])

    ### MELT ###

    solidusFn = solidus_refT + solidus_zCoef * depthFn

    meltFn = fn.branching.conditional([
        (temperatureField > solidusFn, solidusFn),
        (True, temperatureField)
        ])

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

        temperatureField.data[:] = meltFn.evaluate(mesh)

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

    if dirichlet:
        varScales = {'temperatureField': (0., 1.)}
        varBounds = {'temperatureField': (0., 1., '.', '.')}
    else:
        varScales = {}
        varBounds = {}

    blackhole = [0., 0.]

    return Grouper(locals())
