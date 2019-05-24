import underworld as uw
from underworld import function as fn
import math

from planetengine.utilities import Grouper

def build(
    res = 32,
    isoviscous = False,
    aspect = 1.,
    length = 1.,
    Ra = 1e7,
    maxVisc = 3e4,
    tau0 = 4e5,
    tau1 = 1e7,
    ):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### MESH & MESH VARIABLES ###
    
    elementRes = (res, int(int(4. * res * aspect) / 4))
    maxCoord = (aspect, length)

    mesh = uw.mesh.FeMesh_Cartesian(
        elementRes = elementRes,
        maxCoord = maxCoord
        )

    temperatureField = uw.mesh.MeshVariable(mesh, 1)
    temperatureDotField = uw.mesh.MeshVariable(mesh, 1)
    pressureField = uw.mesh.MeshVariable(mesh.subMesh, 1)
    velocityField = uw.mesh.MeshVariable(mesh, 2)

    varsOfState = [((("temperatureField", temperatureField),), ("mesh", mesh))]

    ### BOUNDARIES ###
    
    inner = mesh.specialSets["MinJ_VertexSet"]
    outer = mesh.specialSets["MaxJ_VertexSet"]
    sides = mesh.specialSets["MaxI_VertexSet"] + mesh.specialSets["MinI_VertexSet"]

    velBC = uw.conditions.DirichletCondition(
        variable = velocityField,
        indexSetsPerDof= (sides, inner + outer),
        )

    tempBC = uw.conditions.DirichletCondition(
        variable = temperatureField,
        indexSetsPerDof = (inner + outer,)
        )

    ### RHEOLOGY ###

    invDensityFn = temperatureField * Ra
    buoyancyFn = invDensityFn * (0., 1.)

    if isoviscous:
        viscosityFn = creepViscFn = plasticViscFn = 1.
    else:
        magnitude = fn.math.sqrt(fn.coord()[0]**2 + fn.coord()[1]**2)
        depthFn = mesh.maxCoord[1] - magnitude
        yieldStressFn = tau0 + (tau1 * depthFn)
        secInvFn = fn.tensor.second_invariant(fn.tensor.symmetric(velocityField.fn_gradient))
        plasticViscFn = yieldStressFn / (2. * secInvFn + 1e-18)
        creepViscFn = fn.math.pow(fn.misc.constant(maxVisc), -1. * (temperatureField - 1.))
        viscosityFn = fn.misc.min(maxVisc, fn.misc.max(1., fn.misc.min(creepViscFn, plasticViscFn)))

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
        velocityField = velocityField,
        fn_diffusivity = 1.,
        conditions = [tempBC,]
        )

    ### SOLVING ###

    def solve():
        velocityField.data[:] = 0.
        solver.solve(
            nonLinearIterate = not isoviscous,
            )

    def integrate():
        dt = advDiff.get_max_dt()
        advDiff.integrate(dt)
        return dt

    def iterate():
        solve()
        return integrate()

    ### HOUSEKEEPING: IMPORTANT! ###
    return Grouper(locals())