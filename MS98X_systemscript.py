import underworld as uw
from underworld import function as fn
import math

from planetengine.utilities import Grouper

def build(
    res = 64,
    ratio = 0.54,
    aspect = 1.,
    length = 1.,
    Ra = 1e7,
    heating = 0.,
    surfT = 0.,
    deltaT = 1.,
    diffusivity = 1.,
    buoyancy = 1.,
    creep = 1.,
    creep_sR = 3e4,
    tau = 4e5,
    tau_bR = 26.,
    cont_buoyancy_mR = 1.,
    cont_creep_mR = 1.,
    cont_creep_sR_mR = 1.,
    cont_maxVisc_mR = 1.,
    cont_tau_mR = 1.,
    cont_tau_bR_mR = 1.,
    cont_heating_mR = 1.,
    cont_diffusivity_mR = 1.,
    ):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### MESH & MESH VARIABLES ###

    outerRad = length / (1. - min(0.99999, max(0.00001, ratio)))
    radii = (outerRad - length, outerRad)
    width = length**2 * aspect * 2. / (radii[1]**2 - radii[0]**2)
    midpoint = math.pi / 2.
    angExtentRaw = (midpoint - 0.5 * width, midpoint + 0.5 * width)
    angularExtent = [item * 180. / math.pi for item in angExtentRaw]
    angLen = angExtentRaw[1] - angExtentRaw[0]
    radRes = res
    angRes = 4 * int(angLen * (int(radRes * radii[1] / length)) / 4.)
    elementRes = (radRes, angRes)

    mesh = uw.mesh.FeMesh_Annulus(
        elementRes = elementRes,
        radialLengths = radii,
        angularExtent = angularExtent,
        periodic = [False, False]
        )

    temperatureField = uw.mesh.MeshVariable(mesh, 1)
    temperatureDotField = uw.mesh.MeshVariable(mesh, 1)
    pressureField = uw.mesh.MeshVariable(mesh.subMesh, 1)
    velocityField = uw.mesh.MeshVariable(mesh, 2)

    swarm = uw.swarm.Swarm(mesh = mesh, particleEscape = True)
    swarm.populate_using_layout(uw.swarm.layouts.PerCellSpaceFillerLayout(swarm, 12))
    materialVar = swarm.add_variable(dataType = "int", count = 1)
    repopulator = uw.swarm.PopulationControl(
        swarm,
        aggressive = True,
        splitThreshold = 0.15,
        maxDeletions = 2,
        maxSplits = 10,
        particlesPerCell = 10
        )

    varsOfState = [
        ((("temperatureField", temperatureField),), ("mesh", mesh)),
        ((("materialVar", materialVar),), ("swarm", swarm)),
        ]

    ### BOUNDARIES ###
    
    inner = mesh.specialSets["inner"]
    outer = mesh.specialSets["outer"]
    sides = mesh.specialSets["MaxJ_VertexSet"] + mesh.specialSets["MinJ_VertexSet"]

    velBC = uw.conditions.RotatedDirichletCondition(
        variable = velocityField,
        indexSetsPerDof = (inner + outer, sides),
        basis_vectors = (mesh.bnd_vec_normal, mesh.bnd_vec_tangent)
        )

    tempBC = uw.conditions.DirichletCondition(
        variable = temperatureField,
        indexSetsPerDof = (inner + outer,)
        )

    ### RHEOLOGY ###

    vc = uw.mesh.MeshVariable(mesh = mesh, nodeDofCount = 2)
    vc_eqNum = uw.systems.sle.EqNumber(vc, False )
    vcVec = uw.systems.sle.SolutionVector(vc, vc_eqNum)

    baseT = surfT + deltaT
    magnitude = fn.math.sqrt(fn.coord()[0]**2 + fn.coord()[1]**2)
    depthFn = (mesh.radialLengths[1] - magnitude) / length

    buoyancyFn = temperatureField * Ra * mesh.unitvec_r_Fn * fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: buoyancy,
            1: buoyancy * cont_buoyancy_mR,
            }        
        )

    yieldStressFn = fn.branching.map(
        fn_key = materialVar,
        mapping = {
            0: tau * (1. + (tau_bR - 1) * depthFn),
            1: tau * cont_tau_mR * (1. + (tau_bR * cont_tau_bR_mR - 1) * depthFn)
            }        
        )

    secInvFn = fn.tensor.second_invariant(fn.tensor.symmetric(vc.fn_gradient))

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

    viscosityFn = fn.misc.min(creepViscFn, plasticViscFn)

    # this is just for ease of visualisation:
    viscosityProj = uw.mesh.MeshVariable(mesh, 1)
    viscosityProjector = uw.utils.MeshVariable_Projection(viscosityProj, viscosityFn)

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

    ### SYSTEMS ###

    stokes = uw.systems.Stokes(
        velocityField = velocityField,
        pressureField = pressureField,
        conditions = [velBC,],
        fn_viscosity = viscosityFn,
        fn_bodyforce = buoyancyFn,
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
        velocityField = velocityField,
        order = 2,
        )

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
        viscosityProjector.solve()

    def integrate():
        dt = min(advDiff.get_max_dt(), advector.get_max_dt())
        advDiff.integrate(dt)
        advector.integrate(dt)
        repopulator.repopulate()
        return dt

    def iterate():
        solve()
        return integrate()

    ### HOUSEKEEPING: IMPORTANT! ###
    return Grouper(locals())