import planetengine
import underworld as uw
import numpy as np
from underworld import function as fn

from planetengine.utilities import Grouper
from planetengine import initials as InitialConditions
from planetengine.utilities import CoordSystems

def build(
        pert = 0.2,
        freq = 1.,
        phase = 0.,
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
            InitialConditions.Sinusoidal(
                system.temperatureField.data,
                curvedBox(system.mesh.data),
                phase = phase,
                freq = freq,
                pert = pert,
                tempRange = (system.surfT, system.baseT)
                ),
            InitialConditions.Indices(
                system.temperatureField.data,
                [(system.outer.data, system.surfT),
                (system.inner.data, system.baseT)]
                ),
            InitialConditions.SetVal(
                [system.velocityField.data,
                 system.pressureField.data,
                 system.temperatureDotField.data,
                 system.step.value,
                 system.modeltime.value],
                0.
                ),
            ])

        initialConditions.apply()
        system.step.value = 0
        system.modeltime.value = 0.
        system.solve()

    ### HOUSEKEEPING: IMPORTANT! ###

    return Grouper(locals())