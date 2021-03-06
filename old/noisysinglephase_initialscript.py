import planetengine
import underworld as uw
import numpy as np
from underworld import function as fn

from planetengine.utilities import Grouper
from planetengine import initials as InitialConditions
from planetengine.utilities import CoordSystems

def build(
        gradient = 3.,
        exponent = 0.5,
        smoothness = 10.,
        randomSeed = 1066,
        ):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### INITIALS ###

    def apply(system):

        if type(system.mesh) == uw.mesh._spherical_mesh.FeMesh_Annulus:
            phase = 1.
            curvedBox = CoordSystems.Radial(
                system.mesh.radialLengths,
                system.mesh.angularExtent,
                boxDims = ((0., 1.), (0., 1.))
                ).curved_box
        else:
            def curvedBox(argument):
                return argument

        initialConditions = InitialConditions.Group([
            InitialConditions.NoisyGradient(
                system.temperatureField.data,
                curvedBox(system.mesh.data),
                gradient = gradient,
                exponent = exponent,
                smoothness = smoothness,
                randomSeed = randomSeed,
                tempRange = (system.surfT, system.baseT),
                ),
            InitialConditions.Indices(
                system.temperatureField.data,
                [(system.outer.data, system.surfT),
                (system.inner.data, system.baseT)]
                ),
            InitialConditions.SetVal(
                [system.velocityField.data,
                 system.pressureField.data,
                 system.temperatureDotField.data],
                0.
                ),
            ])

        initialConditions.apply()
        system.step.value = 0
        system.modeltime.value = 0.
        system.solve()

    ### HOUSEKEEPING: IMPORTANT! ###

    return Grouper(locals())