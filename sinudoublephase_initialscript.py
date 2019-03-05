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
        cont_centre = 0.5,
        cont_width = 0.5,
        cont_thickness = 0.035,
        ):

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### INITIALS ###

    def attach(system):

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
                tempRange = (system.surfT, system.baseT),
                ),
            InitialConditions.Indices(
                system.temperatureField.data,
                [(system.outer.data, system.surfT),
                (system.inner.data, system.baseT)]
                ),
            InitialConditions.Extents(
                system.materialVar.data, 
                curvedBox(system.swarm.particleCoordinates.data),
                initialExtents = [
                    (0, fn.misc.constant(True)),
                    (1, fn.shape.Polygon(
                        np.array([
                            [cont_centre - cont_width / 2., 1. - cont_thickness],
                            [cont_centre - cont_width / 2., 1.],
                            [cont_centre + cont_width / 2., 1.],
                            [cont_centre + cont_width / 2., 1. - cont_thickness]
                            ])
                        )),
                    #(2, fn.shape.Polygon(np.array([[0., 0.], [0., 0.02], [2., 0.02], [2., 0.]]))),
                    ]
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

        def apply():
            initialConditions.apply()
            system.solve()

        ### HOUSEKEEPING: IMPORTANT! ###

        group = Grouper(locals())
        group.SetVal('script', script)
        group.SetVal('inputs', inputs)
        return group

    ### HOUSEKEEPING: IMPORTANT! ###

    return Grouper(locals())