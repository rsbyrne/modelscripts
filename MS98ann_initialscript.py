import planetengine
import underworld as uw
from underworld import function as fn

from planetengine.utilities import Grouper
from planetengine import initials as InitialConditions
from planetengine.utilities import CoordSystems

def build(phase = 0.):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### INITIALS ###

    def apply(system):

        if type(system.mesh) == uw.mesh._spherical_mesh.FeMesh_Annulus:

            curvedBox = CoordSystems.Radial(
                system.mesh.radialLengths,
                system.mesh.angularExtent,
                boxDims = ((0., 1.), (0., 1.))
                ).curved_box(
                    system.mesh.data
                    )

            initialConditions = InitialConditions.Group([
                InitialConditions.Sinusoidal(system.temperatureField.data, curvedBox, phase = phase + 1.),
                InitialConditions.Indices(
                    system.temperatureField.data,
                    [(system.outer.data, 0.),
                    (system.inner.data, 1.)]
                    ),
                InitialConditions.SetVal(
                    [system.velocityField.data, system.pressureField.data, system.temperatureDotField.data],
                    0.
                    ),
                ])

        else:

            initialConditions = InitialConditions.Group([
                InitialConditions.Sinusoidal(system.temperatureField.data, system.mesh.data, phase = phase),
                InitialConditions.Indices(
                    system.temperatureField.data,
                    [(system.outer.data, 0.),
                    (system.inner.data, 1.)]
                    ),
                InitialConditions.SetVal(
                    [system.velocityField.data, system.pressureField.data, system.temperatureDotField.data],
                    0.
                    ),
                ])

        initialConditions.apply()
        system.solve()

    ### HOUSEKEEPING: IMPORTANT! ###
    return Grouper(locals())
