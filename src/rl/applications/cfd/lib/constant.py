# This file contains all of the constants used for the open foam cases
# reference values
P_INF = 0.0
U_INF = 1.0
RHO_INF = 1.0
CHORD = 1.0
SPAN = 1.0

#FunkySetBoundaryDict constants
UPPER_SIDE = 'suction_side'  #name of the boundary of the upper side of airfoil in Mesh
LOWER_SIDE = 'pressure_side'  #name of the boundary of the lower sider of airfoil in Mesh
INLET = 'inlet'  #name of boun
OUTLET = 'outlet'
TRAILING_EDGE = 'trailing_edge'
TARGET_AIRFOIL = 'value'
TARGET_INLETOUTLET = 'freestreamValue'
TARGET_INTERNALFIELD = 'internalField'
BOUNDARY_DICT = 'funkySetBoundaryDict'

# transportProperties constants:
TRANSPORT_PROPERTIES_TEMPLATE = 'constant/transportProperties_template'
TRANSPORT_PROPERTIES = 'constant/transportProperties'

# Sample Dict for boundary layer sampling
SAMPLEDICT = 'sampleDict'
SAMPLEDICT_TEMPLATE = 'sampleDict_template'
SAMPLE_HEIGHT = 0.1  # distance normal to surface up to which field values get sampled
SAMPLE_STARTHEIGHTOFFSET = -0.000001
POSTPROCESSING_SAMPLEPATH = 'postProcessing/sampleDict/'

#ForceCoeffs_object constants
FORCE_COEFFS_TEMPLATE = 'forceCoeffs_object_template'
FORCE_COEFFS = 'forceCoeffs_object'
FORCE_COEFFS_FILEPATH = ['postProcessing/forceCoeffs_object/', '/forceCoeffs.dat']
POSTPROCESSING_SURFACE_DIRPATH = 'postProcessing/surface_sampling/'
PRESSURE_COEFFS_FILENAME = ['p_pressure_side.raw', 'p_suction_side.raw']
FRICTION_COEFFS_FILENAME = ['wallShearStress_pressure_side.raw', 'wallShearStress_suction_side.raw']
YPLUS_FIELD_FILENAME = ['yPlus_pressure_side.raw', 'yPlus_suction_side.raw']
YPLUS_SUMMERY_FILEPATH = ['postProcessing/yPlus/', 'yPlus.dat']
RESIDUALS_FILEPATH = ['postProcessing/residuals/', 'residuals.dat']

# Check data:
YPLUS_MAX = 2.0
RESIDUAL_MAX = 1 * 10 ** (-4)
