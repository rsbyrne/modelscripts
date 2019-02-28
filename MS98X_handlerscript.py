import underworld as uw
from underworld import function as fn
import math
import time
import glucifer
import numpy as np
import math

from planetengine.utilities import Grouper
import planetengine.analysis as analysis

def build():

    ### HOUSEKEEPING: IMPORTANT! ###

    inputs = locals().copy()
    script = __file__

    ### FIGURES ###

    def make_figs(system, step, modeltime):
        fig = glucifer.Figure(edgecolour = "white", quality = 2)
        figTempComponent = fig.Surface(
            system.mesh, system.temperatureField,
            colourBar = True
            )
        figVelComponent = fig.VectorArrows(
            system.mesh, system.velocityField
            )
        figViscComponent = fig.Contours(
            system.mesh, fn.math.log10(system.viscosityFn),
            colours = "red black", interval = 0.5, colourBar = False 
            )

        figs = {'fig': fig, }
        return figs

    ### DATA ###

    def make_data(system, step, modeltime):
        zerodDataDict = {
            'Nu': analysis.Analyse.DimensionlessGradient(system.temperatureField, system.mesh,
                surfIndexSet = system.outer, baseIndexSet = system.inner
                ),
            'avTemp': analysis.Analyse.ScalarFieldAverage(system.temperatureField, system.mesh),
            'VRMS': analysis.Analyse.VectorFieldVolRMS(system.velocityField, system.mesh),
            'surfVRMS': analysis.Analyse.VectorFieldSurfRMS(
                system.velocityField, system.mesh, system.outer
                ),
            'avVisc': analysis.Analyse.ScalarFieldAverage(system.viscosityFn, system.mesh),
            'yielding': analysis.Analyse.ScalarFieldAverage(
                fn.branching.conditional([(system.creepViscFn < system.plasticViscFn, 0.), (True, 1.)]),
                system.mesh
                ),
            'step': analysis.Analyse.ArrayStripper(step, (0, 0)),
            'modeltime': analysis.Analyse.ArrayStripper(modeltime, (0, 0)),
            }

        zerodFormatDict = {
            'Nu': "{:.1f}",
            'avTemp': "{:.2f}",
            'VRMS': "{:.2f}",
            'surfVRMS': "{:.2f}",
            'avVisc': "{:.1E}",
            'yielding': "{0:.0%}",
            'step': "{:.0f}",
            'modeltime': "{:.1E}",
            }

        zerodAnalyser = analysis.Analyser('zerodData', zerodDataDict, zerodFormatDict)
        dataCollector = analysis.DataCollector([zerodAnalyser,])
        data = {
            'analysers': [zerodAnalyser,],
            'collectors': [dataCollector,],
            }
        return Grouper(data)

    ### HOUSEKEEPING: IMPORTANT! ###
    return Grouper(locals())
