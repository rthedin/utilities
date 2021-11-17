# script written by Regis Thedin

import os
import sys
import glob
import argparse
from paraview.simple import *

# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ------------------------------------------------------------------------------
# -------------------------------- PARSING INPUTS ------------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--case", "-c",    type=str,  default=os.getcwd(),  help="case full path (default cwd)")
parser.add_argument("--var", "-var",       type=str,  default="U",      help="variable (e.g. U, T, UMean)")
parser.add_argument("--comp", "-comp",     type=str,  default="mag",    help="variable (X, Y, Z, mag only)")
parser.add_argument("--normal", "-normal", type=float,default=80,       help="position of the slice (e.g. 1000 for xNormal.1000, 80 for zNormal.80")
parser.add_argument("--firstFrame", "-ff", type=int, default=0,         help="first frame to be saved")
parser.add_argument("--lastFrame", "-lf",  type=int, default=9999,      help="last frame to be saved (blank for all available)")
parser.add_argument("--sType", "-slice",   type=str, default="zNormal", help="slice type (e.g. zNormal, yNormal, terrain)") 
parser.add_argument("--t0", "-t0",         type=int, default=999  ,     help="time shift for label")
parser.add_argument("--timescale", "-tscale",type=int, default=999,     help="time scale for label")
parser.add_argument("--scaleL", "-lb",     type=float, default=-99,     help="scale lower bound")
parser.add_argument("--scaleU", "-ub",     type=float, default=99,      help="scale upper bound")

args = parser.parse_args()

# Parse inputs
case = args.case
var = args.var
comp = args.comp.upper()
normal = int(args.normal)
anim_fstart = args.firstFrame
anim_fend = args.lastFrame
slicetype = args.sType
t0 = args.t0
timescale = args.timescale
scalelowerbound = args.scaleL
scaleupperbound = args.scaleU

if not os.path.exists(case):
	parser.error("Path does not exist")


# ------------------------------------------------------------------------------
# ----------------------------- ANIMATION PARAMETERS ---------------------------
# ------------------------------------------------------------------------------

# Example call:
# pvbatch --mesa thisscript.py --c '/home/rthedin/OpenFOAM/rthedin-6/run/gravityWaves/26_bigUnifU_stable_z15_RLup5'
#                               -var U
#                               -comp Z
#                               -slice xNormal
#                               -normal 80
#                               -lb -1
#                               -ub 1
#                               -t0 20000
#								-ff 32 
#								-lf 40

# View parameters
renderViewSize = [900, 700]

# Output images parameters 
ratio = 1100/600. # 1100 x 600px is a good ratio for streamwise slices or non-square-ish domain
ratio = 800/600. # 800 x 600px is a good ratio
horiz_resolution = 1100
output_resolution = [horiz_resolution, int(horiz_resolution/ratio)]


# ------------------------------------------------------------------------------
# --------------------- END OF USER-MODIFIABLE PARAMETERS ----------------------
# ------------------------------------------------------------------------------

# Some simple changes for robustness
if var == 'u' or var=='vel':
    var='U'
if comp == 'MAG' or comp=='MAGNITUDE':
    comp='Magnitude'

# Scale for colobar appropriately if no values were given
if var[0]=='U':
    scaleleg = 'U {comp} (m/s)'.format(comp=comp)
    if scalelowerbound==-99 or scaleupperbound==99:
        if comp=='Magnitude' or comp=='X':
            scalelowerbound = 2
            scaleupperbound = 12
        elif comp=='Y' or comp=='Z':
            scalelowerbound = -3.0
            scaleupperbound = 3.0
elif var[0]=='T':
    scaleleg = 'T (K)'
    if scalelowerbound==-99 or scaleupperbound==99:
        scalelowerbound = 300
        scaleupperbound = 320
else:
    scaleleg=var
    if scalelowerbound==-99 or scaleupperbound==99:
        print('\n\nWARNING: Specify lower and upper limits of the scale.\n\n')

# Get list of VTKs
files1 = glob.glob('{case}/sequencedVTK/{var}_{stype}.{h}.*'.format(case=case, var=var, stype=slicetype, h=str(normal).replace('-','m') ));
files1.sort();

# set last frame of animation
if anim_fend==0 or anim_fend>=9999:
	anim_fend = len(files1)

# identify first slice to set the appropriate time shift in label
if t0 == 999:
    fullpath = os.path.realpath(os.path.abspath(files1[0]))
    t0 = int(os.path.basename(os.path.dirname(fullpath)))

# set scale of the time label (one slice represents 100s, 500s, etc). Assumes uniform sampling
if timescale == 999:
    fullpath0 = os.path.realpath(os.path.abspath(files1[0]))
    t0 = int(os.path.basename(os.path.dirname(fullpath0)))
    fullpath1 = os.path.realpath(os.path.abspath(files1[1]))
    t1 = int(os.path.basename(os.path.dirname(fullpath1)))
    timescale = t1-t0

# set output path
if not os.path.exists('{case}/animation'.format(case=case)):
    os.makedirs('{case}/animation'.format(case=case))
anim_outpath = '{case}/animation/{stype}_{h}m_{var}_{comp}.png'.format(case=case, stype=slicetype, h=str(normal).replace('-','m'), var=var, comp=comp)

# Print information
print '---------------- CASE PARAMETERS ----------------'
print '- case:', case
print '- variable:', var
print '- component:', comp
print '- slice type:',slicetype
print '- normal:', normal
print '- time shift:', t0
print '- time scale:', timescale
print '- scale upper bound:', scaleupperbound
print '- scale lower bound:', scalelowerbound
print '- first frame:', anim_fstart
print '- last frame:', anim_fend
print ' '

print 'Reading {var}_{stype}.{normal}.{{{i}..{f}}}.vtk'.format(var=var, stype=slicetype, normal=str(normal).replace('-','m'), i=anim_fstart, f=anim_fend)
print 'Number of available VTKs:', len(files1)
print 'Animation will be saved as', anim_outpath
print ' '

# ------------------------------------------------------------------------------
# --------------------------------- FIRST PANEL --------------------------------
# ------------------------------------------------------------------------------

# ---------------------------------------------------------------- LOAD THE DATA
# create a new 'Legacy VTK Reader'
slices = LegacyVTKReader(FileNames=files1)

# --------------------------------------------------------------- ANIMATION SETUP
# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Set palette
LoadPalette(paletteName='WhiteBackground')

# --------------------------------------------------------------------- HIT OK
# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = renderViewSize

# get layout
layout1 = GetLayout()

# show data in view
slicesDisplay = Show(slices, renderView1)

# trace defaults for the display properties.
slicesDisplay.Representation = 'Surface'
slicesDisplay.ColorArrayName = [None, '']
slicesDisplay.OSPRayScaleArray = var
slicesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
slicesDisplay.SelectOrientationVectors = 'None'
slicesDisplay.ScaleFactor = 3072.0
slicesDisplay.SelectScaleArray = 'None'
slicesDisplay.GlyphType = 'Arrow'
slicesDisplay.GlyphTableIndexArray = 'None'
slicesDisplay.GaussianRadius = 153.6
slicesDisplay.SetScaleArray = ['POINTS', var]
slicesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
slicesDisplay.OpacityArray = ['POINTS', var]
slicesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
slicesDisplay.DataAxesGrid = 'GridAxesRepresentation'
slicesDisplay.SelectionCellLabelFontFile = ''
slicesDisplay.SelectionPointLabelFontFile = ''
slicesDisplay.PolarAxes = 'PolarAxesRepresentation'

# get the material library
materialLibrary1 = GetMaterialLibrary()

# ----------------------------------------------------------- SELECT VARIABLE
if var[0]=='U':
    # set scalar coloring
    ColorBy(slicesDisplay, ('POINTS', var, comp))
else:
    ColorBy(slicesDisplay, ('POINTS', var))

# get color transfer function/color map for 'U'/'T'/'UMean'/...
uLUT = GetColorTransferFunction(var)
uLUT.RGBPoints = [6.088017156980581, 0.231373, 0.298039, 0.752941, 8.98210526138447, 0.865003, 0.865003, 0.865003, 11.876193365788358, 0.705882, 0.0156863, 0.14902]
uLUT.ScalarRangeInitialized = 1.0

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
if scalelowerbound+scaleupperbound != 0:
    uLUT.ApplyPreset('Viridis (matplotlib)', True)

# get opacity transfer function/opacity map for 'U'/'T'/'UMean'/...
uPWF = GetOpacityTransferFunction(var)
uPWF.Points = [6.088017156980581, 0.0, 0.5, 0.0, 11.876193365788358, 1.0, 0.5, 0.0]
uPWF.ScalarRangeInitialized = 1

# --------------------------------------------------------- COLORBAR PROPERTIES
# get color legend/bar for uLUT in view renderView1
uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Title = scaleleg
uLUTColorBar.ComponentTitle = ''
uLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
uLUTColorBar.TitleFontFile = ''
uLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
uLUTColorBar.LabelFontFile = ''

# Rescale transfer function
uLUT.RescaleTransferFunction(scalelowerbound, scaleupperbound)
uPWF.RescaleTransferFunction(scalelowerbound, scaleupperbound)

# Properties modified on uLUTColorBar
uLUTColorBar.RangeLabelFormat = '%-#.1f'
# The fonts will be bigger than what I input here.
# This is a known bug in paraview 5.6: https://gitlab.kitware.com/paraview/paraview/-/issues/19831
uLUTColorBar.TitleFontSize = 5
uLUTColorBar.LabelFontSize = 5
uLUTColorBar.ScalarBarThickness = 10
uLUTColorBar.ScalarBarLength = 0.4  # a third of the height

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(uLUT, slicesDisplay)
slicesDisplay.SetScalarBarVisibility(renderView1, True)
uLUTColorBar.Title = scaleleg
uLUTColorBar.ComponentTitle = ''


# --------------------------------------------------------- CREATE TIME LABEL
# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(Input=slices)

# Properties modified on annotateTimeFilter1
annotateTimeFilter1.Format = 'Time: %.1f s'
annotateTimeFilter1.Shift = t0
annotateTimeFilter1.Scale = timescale

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)

# trace defaults for the display properties.
annotateTimeFilter1Display.Color = [0.0, 0.0, 0.0]
annotateTimeFilter1Display.FontFile = ''
annotateTimeFilter1Display.FontSize = 16

# --------------------------------------------------------- CREATE HEIGHT LABEL
# set active source
SetActiveSource(slices)

# create a new 'Text'
text1 = Text()

# Properties modified on text1
text1.Text = '{stype} = {normal} m'.format(stype=slicetype, normal=normal)

# show data in view
text1Display = Show(text1, renderView1)

# trace defaults for the display properties.
text1Display.Color = [0.0, 0.0, 0.0]
text1Display.FontFile = ''

# Properties modified on text1Display
text1Display.FontSize = 16
text1Display.WindowLocation = 'UpperCenter'

# --------------------------------------------------------- VIEW PROPERTIES
# Properties modified on renderView1
renderView1.EnableOSPRay = 1
renderView1.Shadows = 0
renderView1.CameraParallelProjection = 1
renderView1.ProgressivePasses = 2  # needs --enable-streaming-options

renderView1.CameraFocalPoint = [0, 0, 0]
if slicetype[0] == 'x':
    renderView1.CameraPosition = [-1, 0, 0]
    renderView1.CameraViewUp = [0, 0, 1]
elif slicetype[0] == 'y':
    renderView1.CameraPosition = [0, -1, 0]
    renderView1.CameraViewUp = [0, 0, 1]
elif slicetype[0] == 'z' or slicetype[0:3]=='ter':
    renderView1.CameraPosition = [0, 0, 1]
    renderView1.CameraViewUp = [0, 1, 0]
else:
    print('This slice may require a custom camera view. Edit the script.')

# View parameters
renderView1.InteractionMode = '2D'
renderView1.ResetCamera()

##########################################################################
############################## SAVE ANIMATION ############################
##########################################################################
print('Saving animation...')

# save animation
SaveAnimation(anim_outpath,
	#renderView1,
	layout1, SaveAllViews=1,
    ImageResolution=output_resolution,
    #FontScaling='Do not scale fonts',
    SeparatorWidth=1,
    SeparatorColor=[0.0, 0.0, 0.0],
    #TransparentBackground=1,
    FrameWindow=[anim_fstart, anim_fend])

print('Done.')

