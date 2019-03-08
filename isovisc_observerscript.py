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

    ### PROJECTORS ###

    def make_tools(system):

        tools = {}

        return Grouper(tools)

    ### FIGURES ###

    def make_figs(system, tools):

        fig = glucifer.Figure(edgecolour = "white", quality = 2)

        figTempComponent = fig.Surface(
            system.mesh,
            system.temperatureField,
            colourBar = True
            )

        figVelComponent = fig.VectorArrows(
            system.mesh,
            system.velocityField
            )

        figs = {'fig': fig, }

        return figs

    ### DATA ###

    def make_data(system, tools):

        zerodDataDict = {
            'Nu': analysis.Analyse.DimensionlessGradient(
                system.temperatureField,
                system.mesh,
                system.outer,
                system.inner,
                ),
            'avTemp': analysis.Analyse.ScalarFieldAverage(
                system.temperatureField,
                system.mesh,
                ),
            'VRMS': analysis.Analyse.VectorFieldVolRMS(
                system.velocityField,
                system.mesh,
                ),
            'surfVRMS': analysis.Analyse.VectorFieldSurfRMS(
                system.velocityField,
                system.mesh,
                system.outer,
                ),
            'step': analysis.Analyse.ArrayStripper(
                system.step,
                (0, 0),
                ),
            'modeltime': analysis.Analyse.ArrayStripper(
                system.modeltime,
                (0, 0),
                ),
            }

        zerodFormatDict = {
            'Nu': "{:.1f}",
            'avTemp': "{:.2f}",
            'VRMS': "{:.2f}",
            'surfVRMS': "{:.2f}",
            'step': "{:.0f}",
            'modeltime': "{:.1E}",
            }

        zerodAnalyser = analysis.Analyser('zerodData', zerodDataDict, zerodFormatDict)
        dataCollector = analysis.DataCollector([zerodAnalyser,])
        data = {
            'analysers': [zerodAnalyser,],
            'collectors': [dataCollector,],
            }

        return data

    ### HOUSEKEEPING: IMPORTANT! ###

    return Grouper(locals())