# script written by Regis Thedin

import os
import sys
import glob
import argparse
from paraview.simple import *

# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ------------------------------------------------------------------------------
# ----------------------------- PARSING INPUTS ---------------------------
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--case", "-c", metavar='full/path/to/case',      help="case full path",
					default='/home/rthedin/OpenFOAM/rthedin-6/run/gravityWaves/26_bigUnifU_stable_z15_RLup5')
parser.add_argument("--xmin", "-xmin", type=float, default=-15000,    help="x minimum value")
parser.add_argument("--xmax", "-xmax", type=float, default=15720,     help="x minimum value")
parser.add_argument("--ymin", "-ymin", type=float, default=-5000,     help="y minimum value")
parser.add_argument("--ymax", "-ymax", type=float, default=15160,     help="y maximum value")
parser.add_argument("--zmax", "-zmax", type=float, default=15400,     help="top of the domain value")
parser.add_argument("--spongeWidth", "-sp", type=float, default=5000, help="sponge layer width")
parser.add_argument("--firstFrame", "-ff", type=int, default=0,       help="first frame to be saved")
parser.add_argument("--lastFrame", "-lf", type=int, default=5,        help="last frame to be saved (9999 for all VTKs present)")

args = parser.parse_args()

# Parse inputs
case = args.case
xmin = args.xmin
xmax = args.xmax
ymin = args.ymin
ymax = args.ymax
zmax = args.zmax
spwidth = args.spongeWidth
anim_fstart = args.firstFrame
anim_fend = args.lastFrame

if not os.path.exists(case):
	parser.error("Path does not exist")

# ------------------------------------------------------------------------------
# ----------------------------- ANIMATION PARAMETERS ---------------------------
# ------------------------------------------------------------------------------

# Example call:
# pvbatch --mesa thisscript.py --c '/home/rthedin/OpenFOAM/rthedin-6/run/gravityWaves/26_bigUnifU_stable_z15_RLup5'
# 							    -xmax  15720
# 							    -xmin  -15000
# 							    -ymax  15160
# 							    -ymin  -5000
#							    -zmax  15400
#							    -sp     5000   # sponge width
#								-ff 32
#								-lf 40


zref=0

x = xmax - xmin
y = ymax - ymin
xmid = xmin + (xmax-xmin)/2
ymid = ymin + (ymax-ymin)/2
zmidref = (zmax-zref)/2

# slices to be saved
ynormalslice = 10000
xnormalslice = 5000
terrainheight1 = 200
terrainheight2 = 500

# Scale for colobar
scalelowerbound = -6.0
scaleupperbound = 6.0

# View parameters
renderViewSize = [800, 300]

# Output images parameters 
ratio = 1100/600. # 1100 x 600px is a good ratio
output_resolution = [2000, int(2000./ratio)]


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# END OF USER-MODIFIABLE PARAMETERS

# Get list of VTKs for each of the four views
files1 = glob.glob('{case}/sequencedVTK/U_terrain.{h}.*'.format(case=case,h=terrainheight1));
files2 = glob.glob('{case}/sequencedVTK/U_terrain.{h}.*'.format(case=case,h=terrainheight1));
files3 = glob.glob('{case}/sequencedVTK/U_yNormal.{y}.*'.format(case=case,y=ynormalslice));  
files4 = glob.glob('{case}/sequencedVTK/U_xNormal.{x}.*'.format(case=case,x=xnormalslice));  
files1.sort();  files2.sort(); files3.sort();  files4.sort()

# set last frame of animation
if anim_fend==0 or anim_fend>=9999:
	anim_fend = min(len(files1), len(files2), len(files3), len(files4))

# set output path
if not os.path.exists('{case}/animation'.format(case=case)):
    os.makedirs('{case}/animation'.format(case=case))
#anim_outpath = '/home/rthedin/OpenFOAM/rthedin-6/run/fviews.png'
anim_outpath = '{case}/animation/fourViews_{path}.png'.format(case=case,path=os.path.basename(case))

# Print information
print '---------------- CASE PARAMETERS ----------------'
print '- case:', case
print '- xmin:', xmin
print '- xmax:', xmax
print '- xmin:', ymin
print '- ymax:', ymax
print '- zmax:', zmax
print '- sponge width:', spwidth
print '- first frame:', anim_fstart
print '- last frame:', anim_fend
print ' '

print 'Number of VTK for terrain at height', terrainheight1, ':', len(files1)
print 'Number of VTK for terrain at height', terrainheight2, ':', len(files2)
print 'Number of VTK for y-normal slice at', ynormalslice, ':', len(files3)
print 'Number of VTK for x-normal slice at', xnormalslice, ':', len(files4)
print 'Animation will be saved as', anim_outpath
print ' '






# ------------------------------------------------------------------------------
# --------------------------------- FIRST PANEL --------------------------------
# ------------------------------------------------------------------------------
print('Working on panel 1')

# ---------------------------------------------------------------- LOAD THE DATA
# create a new 'Legacy VTK Reader'
u_terrain2000 = LegacyVTKReader(FileNames=files1)

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

# show data in view
u_terrain2000Display = Show(u_terrain2000, renderView1)

# trace defaults for the display properties.
u_terrain2000Display.Representation = 'Surface'
u_terrain2000Display.ColorArrayName = [None, '']
u_terrain2000Display.OSPRayScaleArray = 'U'
u_terrain2000Display.OSPRayScaleFunction = 'PiecewiseFunction'
u_terrain2000Display.SelectOrientationVectors = 'None'
u_terrain2000Display.ScaleFactor = 3072.0
u_terrain2000Display.SelectScaleArray = 'None'
u_terrain2000Display.GlyphType = 'Arrow'
u_terrain2000Display.GlyphTableIndexArray = 'None'
u_terrain2000Display.GaussianRadius = 153.6
u_terrain2000Display.SetScaleArray = ['POINTS', 'U']
u_terrain2000Display.ScaleTransferFunction = 'PiecewiseFunction'
u_terrain2000Display.OpacityArray = ['POINTS', 'U']
u_terrain2000Display.OpacityTransferFunction = 'PiecewiseFunction'
u_terrain2000Display.DataAxesGrid = 'GridAxesRepresentation'
u_terrain2000Display.SelectionCellLabelFontFile = ''
u_terrain2000Display.SelectionPointLabelFontFile = ''
u_terrain2000Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
u_terrain2000Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
u_terrain2000Display.DataAxesGrid.XTitleFontFile = ''
u_terrain2000Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
u_terrain2000Display.DataAxesGrid.YTitleFontFile = ''
u_terrain2000Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
u_terrain2000Display.DataAxesGrid.ZTitleFontFile = ''
u_terrain2000Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
u_terrain2000Display.DataAxesGrid.XLabelFontFile = ''
u_terrain2000Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
u_terrain2000Display.DataAxesGrid.YLabelFontFile = ''
u_terrain2000Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
u_terrain2000Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
u_terrain2000Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
u_terrain2000Display.PolarAxes.PolarAxisTitleFontFile = ''
u_terrain2000Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
u_terrain2000Display.PolarAxes.PolarAxisLabelFontFile = ''
u_terrain2000Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
u_terrain2000Display.PolarAxes.LastRadialAxisTextFontFile = ''
u_terrain2000Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
u_terrain2000Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# get the material library
materialLibrary1 = GetMaterialLibrary()

# ------------------------------------------------------------------ SELECT U
# set scalar coloring
ColorBy(u_terrain2000Display, ('POINTS', 'U', 'Z'))

# get color transfer function/color map for 'U'
uLUT = GetColorTransferFunction('U')
uLUT.RGBPoints = [6.088017156980581, 0.231373, 0.298039, 0.752941, 8.98210526138447, 0.865003, 0.865003, 0.865003, 11.876193365788358, 0.705882, 0.0156863, 0.14902]
uLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'U'
uPWF = GetOpacityTransferFunction('U')
uPWF.Points = [6.088017156980581, 0.0, 0.5, 0.0, 11.876193365788358, 1.0, 0.5, 0.0]
uPWF.ScalarRangeInitialized = 1

# --------------------------------------------------------- COLORBAR PROPERTIES
# get color legend/bar for uLUT in view renderView1
uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Title = 'U_z (m/s) first'
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
uLUTColorBar.TitleFontSize = 10
uLUTColorBar.LabelFontSize = 10
uLUTColorBar.ScalarBarLength = 0.4  # a third of the height

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(uLUT, u_terrain2000Display)
u_terrain2000Display.SetScalarBarVisibility(renderView1, False)

# --------------------------------------------------------- CREATE TIME LABEL
# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(Input=u_terrain2000)

# Properties modified on annotateTimeFilter1
annotateTimeFilter1.Format = 'Time: %.1f s'
annotateTimeFilter1.Scale = 100.0

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)

# trace defaults for the display properties.
annotateTimeFilter1Display.Color = [0.0, 0.0, 0.0]
annotateTimeFilter1Display.FontFile = ''
annotateTimeFilter1Display.FontSize = 16

# --------------------------------------------------------- CREATE 200m LABEL
# set active source
SetActiveSource(u_terrain2000)

# create a new 'Text'
text1 = Text()

# Properties modified on text1
text1.Text = '{terrainheight1} m'.format(terrainheight1=terrainheight1)

# show data in view
text1Display = Show(text1, renderView1)

# trace defaults for the display properties.
text1Display.Color = [0.0, 0.0, 0.0]
text1Display.FontFile = ''

# Properties modified on text1Display
text1Display.FontSize = 16
text1Display.WindowLocation = 'UpperCenter'

# ------------------------------------------------- CREATE CYLINDER FOR SLICE
# create a new 'Cylinder'
cylinder1 = Cylinder()

# Properties modified on cylinder2
cylinder1.Height = x
cylinder1.Radius = 60.0
cylinder1.Center = [0, 0, 800.0]

# show data in view
cylinder1Display = Show(cylinder1, renderView1)

# trace defaults for the display properties.
cylinder1Display.Representation = 'Surface'
cylinder1Display.ColorArrayName = [None, '']
cylinder1Display.OSPRayScaleArray = 'Normals'
cylinder1Display.OSPRayScaleFunction = 'PiecewiseFunction'
cylinder1Display.SelectOrientationVectors = 'None'
cylinder1Display.ScaleFactor = 3072.0 #1500?
cylinder1Display.SelectScaleArray = 'None'
cylinder1Display.GlyphType = 'Arrow'
cylinder1Display.GlyphTableIndexArray = 'None'
cylinder1Display.GaussianRadius = 75.0
cylinder1Display.SetScaleArray = ['POINTS', 'Normals']
cylinder1Display.ScaleTransferFunction = 'PiecewiseFunction'
cylinder1Display.OpacityArray = ['POINTS', 'Normals']
cylinder1Display.OpacityTransferFunction = 'PiecewiseFunction'
cylinder1Display.DataAxesGrid = 'GridAxesRepresentation'
cylinder1Display.SelectionCellLabelFontFile = ''
cylinder1Display.SelectionPointLabelFontFile = ''
cylinder1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
cylinder1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
cylinder1Display.DataAxesGrid.XTitleFontFile = ''
cylinder1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
cylinder1Display.DataAxesGrid.YTitleFontFile = ''
cylinder1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
cylinder1Display.DataAxesGrid.ZTitleFontFile = ''
cylinder1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
cylinder1Display.DataAxesGrid.XLabelFontFile = ''
cylinder1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
cylinder1Display.DataAxesGrid.YLabelFontFile = ''
cylinder1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
cylinder1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
cylinder1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
cylinder1Display.PolarAxes.PolarAxisTitleFontFile = ''
cylinder1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
cylinder1Display.PolarAxes.PolarAxisLabelFontFile = ''
cylinder1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
cylinder1Display.PolarAxes.LastRadialAxisTextFontFile = ''
cylinder1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
cylinder1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# change solid color - Dark green
cylinder1Display.DiffuseColor = [0.16862745098039217, 0.9215686274509803, 0.0]

# --------------------------------------------------------- ROTATE CYLINDER
# create a new 'Transform'
transform1 = Transform(Input=cylinder1)
transform1.Transform = 'Transform'

# Properties modified on transform1.Transform
transform1.Transform.Rotate = [0.0, 0.0, 90.0]
transform1.Transform.Translate = [xmid, ynormalslice, 0.0]

# show data in view
transform1Display = Show(transform1, renderView1)

# trace defaults for the display properties.
transform1Display.Representation = 'Surface'
transform1Display.ColorArrayName = [None, '']
transform1Display.OSPRayScaleArray = 'Normals'
transform1Display.OSPRayScaleFunction = 'PiecewiseFunction'
transform1Display.SelectOrientationVectors = 'None'
transform1Display.ScaleFactor = 3072.0
transform1Display.SelectScaleArray = 'None'
transform1Display.GlyphType = 'Arrow'
transform1Display.GlyphTableIndexArray = 'None'
transform1Display.GaussianRadius = 153.6
transform1Display.SetScaleArray = ['POINTS', 'Normals']
transform1Display.ScaleTransferFunction = 'PiecewiseFunction'
transform1Display.OpacityArray = ['POINTS', 'Normals']
transform1Display.OpacityTransferFunction = 'PiecewiseFunction'
transform1Display.DataAxesGrid = 'GridAxesRepresentation'
transform1Display.SelectionCellLabelFontFile = ''
transform1Display.SelectionPointLabelFontFile = ''
transform1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
transform1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
transform1Display.DataAxesGrid.XTitleFontFile = ''
transform1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
transform1Display.DataAxesGrid.YTitleFontFile = ''
transform1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
transform1Display.DataAxesGrid.ZTitleFontFile = ''
transform1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
transform1Display.DataAxesGrid.XLabelFontFile = ''
transform1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
transform1Display.DataAxesGrid.YLabelFontFile = ''
transform1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
transform1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
transform1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
transform1Display.PolarAxes.PolarAxisTitleFontFile = ''
transform1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
transform1Display.PolarAxes.PolarAxisLabelFontFile = ''
transform1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
transform1Display.PolarAxes.LastRadialAxisTextFontFile = ''
transform1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
transform1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(cylinder1, renderView1)

# change solid color
transform1Display.DiffuseColor = [0.16862745098039217, 0.9215686274509803, 0.0]

# --------------------------------------------------------- VIEW PROPERTIES
# Properties modified on renderView1
renderView1.EnableOSPRay = 1
renderView1.Shadows = 1
renderView1.CameraParallelProjection = 1
renderView1.ProgressivePasses = 2  # needs --enable-streaming-options

# View parameters
renderView1.InteractionMode = '2D'
renderView1.CameraFocalPoint = [xmid, ymid, 5000]  # if z is high enough, the image doesn't change
renderView1.CameraPosition = [xmid, ymid, 15000] # 8, 10, 15k did not change the image. No image at 5k
renderView1.CameraViewUp = [0.0, 1.0, 0.0]   # we gave the direction, now what orientation. Right now the view could rotate
renderView1.CameraParallelScale = 11000 # the lower, the closer it is

renderView1.Update()
















# ------------------------------------------------------------------------------
# -------------------------------- SECOND PANEL --------------------------------
# ------------------------------------------------------------------------------
print('Working on panel 2')

# ---------------------------------------------------------------- SPLIT LAYOUT
# Getlayout and split
layout1 = GetLayout()
layout1.SplitHorizontal(0, 0.5)  # two side-by-side

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [845, 803]
renderView2.AnnotationColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView2.StereoType = 0
renderView2.Background = [1, 1, 1] # white
renderView2.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView2.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.XTitleFontFile = ''
renderView2.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.YTitleFontFile = ''
renderView2.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.ZTitleFontFile = ''
renderView2.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.XLabelFontFile = ''
renderView2.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.YLabelFontFile = ''
renderView2.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
renderView2.AxesGrid.ZLabelFontFile = ''

# place view in the layout
layout1.AssignView(2, renderView2)

# ---------------------------------------------------------------- LOAD THE DATA
# create a new 'Legacy VTK Reader'
u_terrain5000 = LegacyVTKReader(FileNames=files2)

# show data in view
u_terrain5000Display = Show(u_terrain5000, renderView2)

# trace defaults for the display properties.
u_terrain5000Display.Representation = 'Surface'
u_terrain5000Display.ColorArrayName = [None, '']
u_terrain5000Display.OSPRayScaleArray = 'U'
u_terrain5000Display.OSPRayScaleFunction = 'PiecewiseFunction'
u_terrain5000Display.SelectOrientationVectors = 'None'
u_terrain5000Display.ScaleFactor = 3072.0
u_terrain5000Display.SelectScaleArray = 'None'
u_terrain5000Display.GlyphType = 'Arrow'
u_terrain5000Display.GlyphTableIndexArray = 'None'
u_terrain5000Display.GaussianRadius = 153.6
u_terrain5000Display.SetScaleArray = ['POINTS', 'U']
u_terrain5000Display.ScaleTransferFunction = 'PiecewiseFunction'
u_terrain5000Display.OpacityArray = ['POINTS', 'U']
u_terrain5000Display.OpacityTransferFunction = 'PiecewiseFunction'
u_terrain5000Display.DataAxesGrid = 'GridAxesRepresentation'
u_terrain5000Display.SelectionCellLabelFontFile = ''
u_terrain5000Display.SelectionPointLabelFontFile = ''
u_terrain5000Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
u_terrain5000Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
u_terrain5000Display.DataAxesGrid.XTitleFontFile = ''
u_terrain5000Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
u_terrain5000Display.DataAxesGrid.YTitleFontFile = ''
u_terrain5000Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
u_terrain5000Display.DataAxesGrid.ZTitleFontFile = ''
u_terrain5000Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
u_terrain5000Display.DataAxesGrid.XLabelFontFile = ''
u_terrain5000Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
u_terrain5000Display.DataAxesGrid.YLabelFontFile = ''
u_terrain5000Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
u_terrain5000Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
u_terrain5000Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
u_terrain5000Display.PolarAxes.PolarAxisTitleFontFile = ''
u_terrain5000Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
u_terrain5000Display.PolarAxes.PolarAxisLabelFontFile = ''
u_terrain5000Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
u_terrain5000Display.PolarAxes.LastRadialAxisTextFontFile = ''
u_terrain5000Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
u_terrain5000Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# ------------------------------------------------------------------ SELECT U
# set scalar coloring
ColorBy(u_terrain5000Display, ('POINTS', 'U', 'Z'))

# --------------------------------------------------------- COLORBAR PROPERTIES
# get color legend/bar for uLUT in view renderView2
uLUTColorBar = GetScalarBar(uLUT, renderView2)
uLUTColorBar.Title = 'U_z (m/s) second'
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
uLUTColorBar.TitleFontSize = 10
uLUTColorBar.LabelFontSize = 10
uLUTColorBar.ScalarBarLength = 0.4  # a third of the height

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(uLUT, u_terrain5000Display)
u_terrain5000Display.SetScalarBarVisibility(renderView2, False)

# --------------------------------------------------------- CREATE 200m LABEL
# set active source
SetActiveSource(u_terrain5000)

# create a new 'Text'
text2 = Text()

# Properties modified on text1
text2.Text = '{terrainheight2} m'.format(terrainheight2=terrainheight2)

# show data in view
text2Display = Show(text2, renderView2)

# trace defaults for the display properties.
text2Display.Color = [0.0, 0.0, 0.0]
text2Display.FontFile = ''

# Properties modified on text1Display
text2Display.FontSize = 16
text2Display.WindowLocation = 'UpperCenter'

# ------------------------------------------------- CREATE CYLINDER FOR SLICE
# create a new 'Cylinder'
cylinder2 = Cylinder()

# Properties modified on cylinder1
cylinder2.Height = y
cylinder2.Radius = 60.0
cylinder2.Center = [xnormalslice, ymid, 800.0]

# show data in view
cylinder2Display = Show(cylinder2, renderView2)

# trace defaults for the display properties.
cylinder2Display.Representation = 'Surface'
cylinder2Display.ColorArrayName = [None, '']
cylinder2Display.OSPRayScaleArray = 'Normals'
cylinder2Display.OSPRayScaleFunction = 'PiecewiseFunction'
cylinder2Display.SelectOrientationVectors = 'None'
cylinder2Display.ScaleFactor = 60.0
cylinder2Display.SelectScaleArray = 'None'
cylinder2Display.GlyphType = 'Arrow'
cylinder2Display.GlyphTableIndexArray = 'None'
cylinder2Display.GaussianRadius = 3.0
cylinder2Display.SetScaleArray = ['POINTS', 'Normals']
cylinder2Display.ScaleTransferFunction = 'PiecewiseFunction'
cylinder2Display.OpacityArray = ['POINTS', 'Normals']
cylinder2Display.OpacityTransferFunction = 'PiecewiseFunction'
cylinder2Display.DataAxesGrid = 'GridAxesRepresentation'
cylinder2Display.SelectionCellLabelFontFile = ''
cylinder2Display.SelectionPointLabelFontFile = ''
cylinder2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
cylinder2Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
cylinder2Display.DataAxesGrid.XTitleFontFile = ''
cylinder2Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
cylinder2Display.DataAxesGrid.YTitleFontFile = ''
cylinder2Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
cylinder2Display.DataAxesGrid.ZTitleFontFile = ''
cylinder2Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
cylinder2Display.DataAxesGrid.XLabelFontFile = ''
cylinder2Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
cylinder2Display.DataAxesGrid.YLabelFontFile = ''
cylinder2Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
cylinder2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
cylinder2Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
cylinder2Display.PolarAxes.PolarAxisTitleFontFile = ''
cylinder2Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
cylinder2Display.PolarAxes.PolarAxisLabelFontFile = ''
cylinder2Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
cylinder2Display.PolarAxes.LastRadialAxisTextFontFile = ''
cylinder2Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
cylinder2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# change solid color - Dark yellow
cylinder2Display.DiffuseColor = [0.9450980, 0.88235294, 0.172549019]

# --------------------------------------------------------- VIEW PROPERTIES
# Properties modified on renderView2
renderView2.EnableOSPRay = 1
renderView2.Shadows = 1

# View parameters
renderView2.InteractionMode = '2D'
renderView2.CameraFocalPoint = [xmid, ymid, 5000]  # if z is high enough, the image doesn't change
renderView2.CameraPosition = [xmid, ymid, 15000] # 8, 10, 15k did not change the image. No image at 5k
renderView2.CameraViewUp = [0.0, 1.0, 0.0]   # from top z
renderView2.CameraParallelScale = 11000 # the lower, the closer it is

renderView2.Update()























# ------------------------------------------------------------------------------
# --------------------------------- THIRD PANEL --------------------------------
# ------------------------------------------------------------------------------
print('Working on panel 3')

# ---------------------------------------------------------------- SPLIT LAYOUT
# set active view
SetActiveView(renderView1)

# split cell
layout1.SplitVertical(1, 0.5) # one under the first of a side-by-side (0.5 = 50%)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [845, 386]
renderView3.AnnotationColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView3.StereoType = 0
renderView3.Background = [1, 1, 1] # white
renderView3.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView3.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.XTitleFontFile = ''
renderView3.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.YTitleFontFile = ''
renderView3.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.ZTitleFontFile = ''
renderView3.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.XLabelFontFile = ''
renderView3.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.YLabelFontFile = ''
renderView3.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
renderView3.AxesGrid.ZLabelFontFile = ''

# place view in the layout
layout1.AssignView(4, renderView3)   # why 4? shouln't be 3?

# ---------------------------------------------------------------- LOAD THE DATA
# create a new 'Legacy VTK Reader'
#u_yNormal100000 =  LegacyVTKReader(FileNames=['/home/rthedin/OpenFOAM/rthedin-6/run/gravityWaves/26_bigUnifU_stable_z15_RLup5/sequencedVTK/U_yNormal.10000.0001.vtk', '/home/rthedin/OpenFOAM/rthedin-6/run/gravityWaves/26_bigUnifU_stable_z15_RLup5/sequencedVTK/U_yNormal.10000.0002.vtk'])
u_yNormal100000 = LegacyVTKReader(FileNames=files3)

# show data in view
u_yNormal100000Display = Show(u_yNormal100000, renderView3)

# trace defaults for the display properties.
u_yNormal100000Display.Representation = 'Surface'
u_yNormal100000Display.ColorArrayName = [None, '']
u_yNormal100000Display.OSPRayScaleArray = 'U'
u_yNormal100000Display.OSPRayScaleFunction = 'PiecewiseFunction'
u_yNormal100000Display.SelectOrientationVectors = 'None'
u_yNormal100000Display.ScaleFactor = 3072.0
u_yNormal100000Display.SelectScaleArray = 'None'
u_yNormal100000Display.GlyphType = 'Arrow'
u_yNormal100000Display.GlyphTableIndexArray = 'None'
u_yNormal100000Display.GaussianRadius = 153.6
u_yNormal100000Display.SetScaleArray = ['POINTS', 'U']
u_yNormal100000Display.ScaleTransferFunction = 'PiecewiseFunction'
u_yNormal100000Display.OpacityArray = ['POINTS', 'U']
u_yNormal100000Display.OpacityTransferFunction = 'PiecewiseFunction'
u_yNormal100000Display.DataAxesGrid = 'GridAxesRepresentation'
u_yNormal100000Display.SelectionCellLabelFontFile = ''
u_yNormal100000Display.SelectionPointLabelFontFile = ''
u_yNormal100000Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
u_yNormal100000Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.DataAxesGrid.XTitleFontFile = ''
u_yNormal100000Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.DataAxesGrid.YTitleFontFile = ''
u_yNormal100000Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.DataAxesGrid.ZTitleFontFile = ''
u_yNormal100000Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.DataAxesGrid.XLabelFontFile = ''
u_yNormal100000Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.DataAxesGrid.YLabelFontFile = ''
u_yNormal100000Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
u_yNormal100000Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.PolarAxes.PolarAxisTitleFontFile = ''
u_yNormal100000Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.PolarAxes.PolarAxisLabelFontFile = ''
u_yNormal100000Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.PolarAxes.LastRadialAxisTextFontFile = ''
u_yNormal100000Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
u_yNormal100000Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# ------------------------------------------------------------------ SELECT U
# set scalar coloring
ColorBy(u_yNormal100000Display, ('POINTS', 'U', 'Z'))

# --------------------------------------------------------- COLORBAR PROPERTIES
# get color legend/bar for uLUT in view renderView3
uLUTColorBar = GetScalarBar(uLUT, renderView3)
uLUTColorBar.Title = 'U_z (m/s) third'
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
uLUTColorBar.TitleFontSize = 10
uLUTColorBar.LabelFontSize = 10
uLUTColorBar.ScalarBarLength = 0.4  # a third of the height

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(uLUT, u_yNormal100000Display)
u_yNormal100000Display.SetScalarBarVisibility(renderView3, False)

# ------------------------------------------------- BOX FOR LAYER POSITION
# create a new 'Box'
box1 = Box()

# Properties modified on box1
box1.XLength = x
box1.YLength = 10
box1.ZLength = spwidth
box1.Center = [xmid, ymin-1000, zmax-spwidth/2]

# show data in view
box1Display = Show(box1, renderView3)

# trace defaults for the display properties.
box1Display.Representation = 'Outline'
box1Display.ColorArrayName = [None, '']
box1Display.OSPRayScaleArray = 'Normals'
box1Display.OSPRayScaleFunction = 'PiecewiseFunction'
box1Display.SelectOrientationVectors = 'None'
box1Display.ScaleFactor = 3072.0
box1Display.SelectScaleArray = 'None'
box1Display.GlyphType = 'Arrow'
box1Display.GlyphTableIndexArray = 'None'
box1Display.GaussianRadius = 153.6
box1Display.SetScaleArray = ['POINTS', 'Normals']
box1Display.ScaleTransferFunction = 'PiecewiseFunction'
box1Display.OpacityArray = ['POINTS', 'Normals']
box1Display.OpacityTransferFunction = 'PiecewiseFunction'
box1Display.DataAxesGrid = 'GridAxesRepresentation'
box1Display.SelectionCellLabelFontFile = ''
box1Display.SelectionPointLabelFontFile = ''
box1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
box1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
box1Display.DataAxesGrid.XTitleFontFile = ''
box1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
box1Display.DataAxesGrid.YTitleFontFile = ''
box1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
box1Display.DataAxesGrid.ZTitleFontFile = ''
box1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
box1Display.DataAxesGrid.XLabelFontFile = ''
box1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
box1Display.DataAxesGrid.YLabelFontFile = ''
box1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
box1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
box1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
box1Display.PolarAxes.PolarAxisTitleFontFile = ''
box1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
box1Display.PolarAxes.PolarAxisLabelFontFile = ''
box1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
box1Display.PolarAxes.LastRadialAxisTextFontFile = ''
box1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
box1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# change solid color
box1Display.AmbientColor = [0.0, 0.0, 0.0]

# --------------------------------------------------------- VIEW PROPERTIES
# Properties modified on renderView3
renderView3.EnableOSPRay = 1
renderView3.Shadows = 1

# current camera placement for renderView3
renderView3.InteractionMode = '2D'
renderView3.CameraPosition = [xmid, -70000, 7567] # -76616,  7567
renderView3.CameraFocalPoint = [xmid, -10000.0, 7567] # -10000.0, 7567]
renderView3.CameraViewUp = [0.0, 0.0, 1.0]
renderView3.CameraParallelScale = 9000  # the lower, the closer it is

# update the view to ensure updated data information
renderView3.Update()



















# # ------------------------------------------------------------------------------
# # -------------------------------- FOURTH PANEL --------------------------------
# # ------------------------------------------------------------------------------
print('Working on panel 4')

# ---------------------------------------------------------------- SPLIT LAYOUT
# set active view
SetActiveView(renderView2)

# split cell
layout1.SplitVertical(2, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.ViewSize = [845, 386]
renderView4.AnnotationColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView4.StereoType = 0
renderView4.Background = [1, 1, 1] # white
renderView4.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView4.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.XTitleFontFile = ''
renderView4.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.YTitleFontFile = ''
renderView4.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.ZTitleFontFile = ''
renderView4.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.XLabelFontFile = ''
renderView4.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.YLabelFontFile = ''
renderView4.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
renderView4.AxesGrid.ZLabelFontFile = ''

# place view in the layout
layout1.AssignView(6, renderView4)

# ---------------------------------------------------------------- LOAD THE DATA
# create a new 'Legacy VTK Reader'
u_xNormal50000 = LegacyVTKReader(FileNames=files4)

# show data in view
u_xNormal50000Display = Show(u_xNormal50000, renderView4)

# trace defaults for the display properties.
u_xNormal50000Display.Representation = 'Surface'
u_xNormal50000Display.ColorArrayName = [None, '']
u_xNormal50000Display.OSPRayScaleArray = 'U'
u_xNormal50000Display.OSPRayScaleFunction = 'PiecewiseFunction'
u_xNormal50000Display.SelectOrientationVectors = 'None'
u_xNormal50000Display.ScaleFactor = 2016.0
u_xNormal50000Display.SelectScaleArray = 'None'
u_xNormal50000Display.GlyphType = 'Arrow'
u_xNormal50000Display.GlyphTableIndexArray = 'None'
u_xNormal50000Display.GaussianRadius = 100.8
u_xNormal50000Display.SetScaleArray = ['POINTS', 'U']
u_xNormal50000Display.ScaleTransferFunction = 'PiecewiseFunction'
u_xNormal50000Display.OpacityArray = ['POINTS', 'U']
u_xNormal50000Display.OpacityTransferFunction = 'PiecewiseFunction'
u_xNormal50000Display.DataAxesGrid = 'GridAxesRepresentation'
u_xNormal50000Display.SelectionCellLabelFontFile = ''
u_xNormal50000Display.SelectionPointLabelFontFile = ''
u_xNormal50000Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
u_xNormal50000Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.DataAxesGrid.XTitleFontFile = ''
u_xNormal50000Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.DataAxesGrid.YTitleFontFile = ''
u_xNormal50000Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.DataAxesGrid.ZTitleFontFile = ''
u_xNormal50000Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.DataAxesGrid.XLabelFontFile = ''
u_xNormal50000Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.DataAxesGrid.YLabelFontFile = ''
u_xNormal50000Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
u_xNormal50000Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.PolarAxes.PolarAxisTitleFontFile = ''
u_xNormal50000Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.PolarAxes.PolarAxisLabelFontFile = ''
u_xNormal50000Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.PolarAxes.LastRadialAxisTextFontFile = ''
u_xNormal50000Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
u_xNormal50000Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# ------------------------------------------------------------------ SELECT U
# set scalar coloring
ColorBy(u_xNormal50000Display, ('POINTS', 'U', 'Z'))

# --------------------------------------------------------- RESCALE COLORBAR
# get color legend/bar for uLUT in view renderView4
uLUTColorBar = GetScalarBar(uLUT, renderView4)
uLUTColorBar.Title = 'U_z (m/s)'
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
uLUTColorBar.TitleFontSize = 10
uLUTColorBar.LabelFontSize = 10
uLUTColorBar.WindowLocation = 'AnyLocation'
uLUTColorBar.Position = [ 0.87, 0.1]
uLUTColorBar.ScalarBarLength = 0.80  # 90% of the height

# show/hide colorbar
u_xNormal50000Display.SetScalarBarVisibility(renderView4, True)

# ------------------------------------------------- BOX FOR LAYER POSITION
# create a new 'Box'
box2 = Box()

# Properties modified on box2
box2.XLength = x
box2.YLength = y
box2.ZLength = spwidth
box2.Center = [xmid, ymid, zmax-spwidth/2]
# show data in view
box2Display = Show(box2, renderView4)

# trace defaults for the display properties.
box2Display.Representation = 'Outline'
box2Display.ColorArrayName = [None, '']
box2Display.OSPRayScaleArray = 'Normals'
box2Display.OSPRayScaleFunction = 'PiecewiseFunction'
box2Display.SelectOrientationVectors = 'None'
box2Display.ScaleFactor = 2016.0
box2Display.SelectScaleArray = 'None'
box2Display.GlyphType = 'Arrow'
box2Display.GlyphTableIndexArray = 'None'
box2Display.GaussianRadius = 100.8
box2Display.SetScaleArray = ['POINTS', 'Normals']
box2Display.ScaleTransferFunction = 'PiecewiseFunction'
box2Display.OpacityArray = ['POINTS', 'Normals']
box2Display.OpacityTransferFunction = 'PiecewiseFunction'
box2Display.DataAxesGrid = 'GridAxesRepresentation'
box2Display.SelectionCellLabelFontFile = ''
box2Display.SelectionPointLabelFontFile = ''
box2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
box2Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
box2Display.DataAxesGrid.XTitleFontFile = ''
box2Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
box2Display.DataAxesGrid.YTitleFontFile = ''
box2Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
box2Display.DataAxesGrid.ZTitleFontFile = ''
box2Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
box2Display.DataAxesGrid.XLabelFontFile = ''
box2Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
box2Display.DataAxesGrid.YLabelFontFile = ''
box2Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
box2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
box2Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
box2Display.PolarAxes.PolarAxisTitleFontFile = ''
box2Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
box2Display.PolarAxes.PolarAxisLabelFontFile = ''
box2Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
box2Display.PolarAxes.LastRadialAxisTextFontFile = ''
box2Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
box2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# change solid color
box2Display.AmbientColor = [0.0, 0.0, 0.0]

# --------------------------------------------------------- VIEW PROPERTIES
# Properties modified on renderView4
renderView4.CameraParallelProjection = 1
renderView4.EnableOSPRay = 1

renderView4.InteractionMode = '2D'
renderView4.CameraPosition = [xmin, ymid, 7567.9609069825]
renderView4.CameraFocalPoint = [-5000, ymid, 7567.9609069825]
renderView4.CameraViewUp = [0.0, 0.0, 1.0]
renderView4.CameraParallelScale = 9000  # the lower, the closer it is

renderView4.Update()







# --------------------------------------------------------- SPLIT PROPERTIES

layout1.SetSplitFraction(1,0.55) # column 1, the top view has 60% of the height
layout1.SetSplitFraction(2,0.55) # column 2, the top view has 60% of the height
#layout1.SetSplitFraction(0,0.60) # vertical split, the first column has 60% of the width



##########################################################################
############################## SAVE ANIMATION ############################
##########################################################################
print('Saving animation')

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


