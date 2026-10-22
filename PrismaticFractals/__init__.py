"""
Light-bounce image / video generator.

The scene is made of shapes - "obstacles" - drawn in a preview image of at most
512x512 pixels that shows the part of the world the selected framing (camera)
looks at; the default framing spans [-2, 2] in both directions. Rays start at
the centre of every pixel inside a firing shape, whatever resolution is later
exported. There is no other
kind of boundary: the app starts with a single obstacle - the circle of radius 2,
filling the default framing, which reflects rays and fires them inside itself - and
everything else is built the same way, by editing, adding or deleting
obstacles.

The figure
  The "figure" is the mathematical result of perfectly simulating the rules
  below - like the Mandelbrot set or the Buddhabrot are the results of their
  own rules. It can never be generated whole; this program generates views of
  it, detailed and accurate enough to learn about it. What defines the figure
  is the termination limits, the segment length, the emission direction, the
  confinement rule, the fuzzing and the obstacles.
  Coloration, decorations and the Buddha blur are not part of it: they only
  process or decorate the data generated from it.

Path engines
  What makes the path of points that each pixel of the figure gets is the "path
  engine", chosen in the Simulation tab. Ray Bouncing (the default) is what this
  description is mostly about: a ray from every pixel bounces around the
  obstacles. Mandelbrot makes the orbit of z -> z^p + c instead (any real
  exponent p; z_0 = 0 or c; the pixel is c, or - for a Julia set - z_0, with a
  Julia constant of your choice; an escaping orbit is clipped at the escape
  radius or ends at the point beyond it). The obstacles belong to neither
  engine: they decide which pixels get a path (their "fire rays inside"), and an
  orbit meets them like a ray does - every segment from one point to the next
  is tested, an edge can pass the orbit through or end it (Reflect is not
  offered). Everything else is shared: the termination limits, the confinement
  rule, the colorations (the metrics and points a coloration can choose are
  those the engine offers: iteration count, path length, last direction,
  termination reason, the four distances and the seven points of the path), the
  Buddha coloration (with the choice of which kinds of path - escaping,
  non-escaping, ended by an obstacle - a channel scatters) and the exports.
  Switching engine resets what the new engine does not accept and says what it
  reset, with the choice to go back; the Emissions tab and fuzzing, which
  Mandelbrot makes no use of, are greyed out, not removed. The segment length
  multiplier is two settings: "after" (both engines) and "before" (Mandelbrot
  only, it scales each step as soon as it is calculated).

Obstacles
  Any number of shapes, each one of eighteen kinds with its own size/angle
  parameters and rotation:

  * an ellipse (width and height 0 to 4, plus rotation; 4.0 x 4.0 is the circle
    that fills the default world),
  * a regular polygon (number of sides and a radius),
  * a Moss's egg, a circular sector, a circular segment, a two-circle lens,
    a Reuleaux polygon (odd number of sides), or a capsule,
  * and four concave shapes made of circular arcs: a trefoil and a quatrefoil
    (three and four overlapping lobes), a lune (a disc with a bite taken out
    by a second disc) and an arbelos (a half-disc minus two smaller ones),
  * two squircles: the superelliptical one (|x/a|^n + |y/b|^n < 1, exponent
    n from 1 to 16) and the Fernández-Guasti one (a squareness from a circle
    to a square),
  * and an Archimedean spiral: a ribbon winding out from the centre to an
    adjustable maximum radius, with an adjustable pitch (gap between turns)
    and ribbon width (a fraction of the pitch),
  * and the snail of Pythagoras (the spiral of Theodorus): a chosen number of
    right triangles, with legs 1 and sqrt(k), laid round a common corner,
  * and the Hügelschäffer egg (half-length, half-width and a shift that
    makes one end more pointed),
  * and the sine flower (a circle whose radius is raised by a sine of the
    angle round the centre: a whole number of petals and an amplitude).

  Each is placed by a polar position (r, theta) from the origin plus an x/y
  offset, and has its own incoming and outgoing ray effects and its own "Fire
  rays inside" box. Every one of these fields can be animated by keyframes.
  An obstacle's kind is chosen when it is made, with the "new..." menu in the
  Obstacles tab, and cannot be changed afterwards (delete it and make
  another); it has only the parameters its own kind needs.

Ray effects
  What happens to a ray as it reaches the edge of a shape depends on which
  way it is crossing: coming in from outside uses the shape's "incoming ray
  effect", going out from inside its "outgoing ray effect". Each is one of

    Reflect       a mirror (using the boundary's fuzzing settings),
    Terminate     the ray ends the instant it reaches the edge,
    Pass through  the ray carries on across the edge. This is neither a bounce
                  nor a recorded path point, and it doesn't cut the segment
                  short: the ray simply becomes inside (or outside) that shape.

  Reflect both ways makes a container that rays never leave;
  Pass through both ways makes the shape invisible to rays; incoming Pass
  through with outgoing Reflect lets rays in but never out. A new obstacle
  defaults to Reflect both ways, with "fire rays inside" on.

Rays
  A ray starts at the centre of every pixel that is inside ANY shape whose
  "Fire rays inside" box is ticked. Being inside a shape whose box is clear
  makes no difference, and with no shape firing there are no rays at all. The
  ray's direction is set by the emission settings (below) - by default up and
  to the right, at 45 degrees - and it starts out inside every shape that
  contains its start point. It reflects, terminates or passes through at each shape's
  edge as described above, and it reflects off any earlier segment of its own
  path. A ray terminates when

  * it becomes confined (see below),
  * it has bounced as many times as the iteration limit allows (1 to 9999),
  * its total path length reaches the distance limit (in units, where one unit
    is half the image width; 0, the default, means no distance limit). If the
    limit is reached part-way along a segment, the segment is cut off there,
  * it reaches an edge whose effect for that direction is Terminate,
  * it strikes a reflective surface at exactly perpendicular incidence (which
    side it would retrace along is undefined, so it terminates there), or
  * it escapes: nothing is left ahead for it to hit, so its last point is
    where it is the escape radius (a setting; default 4) from the origin -
    or the last edge it passed through, if that is further out.

  Emission direction: each ray starts out pointing either straight up (90
  degrees) or from its own start pixel towards a target point, which can be
  anywhere (a pixel exactly on it points up) and is positioned the way an
  obstacle is: a distance r at angle theta from the origin, plus an x and a y
  offset; the angle offset is then added,
  positive meaning more counterclockwise, and fuzzing's emission jitter (if on)
  on top. The defaults - Up, offset -45 degrees - give the up-and-to-the-right
  direction of 45 degrees.

  Confinement is judged on the last few points of the ray's path (the lookback,
  4 by default; 2 to 999). With the Box method (the default) the ray is confined
  when those points fit in a box smaller than the threshold in both directions;
  with Path length, when the segments between them add up to less than the
  threshold. The threshold is one setting serving as either, in world units (the
  image spans -2 to 2), and can be any number above zero; its default, one pixel
  of the 512-pixel preview (4/512), is what was always built in (2/512 while the world was half the size).

  A segment length multiplier (0.01 to 1; 1 means no effect) shortens every
  ordinary bounce's segment to that fraction of its true length, after which
  the ray continues in the newly bounced direction from where it ended. A
  segment includes any edges the ray passed through on the way.

Fuzzing
  Optional randomness/offsets applied to three kinds of angle: the ray's
  emission direction (a random jitter added once, when it leaves its start
  pixel), the deflection angle of each bounce off any reflecting edge, and the
  deflection angle of each bounce off the ray's own earlier
  path (each of the latter two with an offset, a jitter range and a
  multiplier). A seed makes the randomness reproducible. The whole tab is
  greyed out while fuzzing is off.

Coloration
  For every ray, thirteen metrics are measured (each scaled to 0..1): four
  that compare two chosen sample points (horizontal, vertical, Manhattan and
  Euclidean distance) and nine others (bounce counts and fractions, loop area,
  bounce angles, path length, last-segment direction and length, and why the
  ray terminated - one of six reasons, spaced evenly from 0 to 1). Fifteen sample points can be chosen: thirteen picked out of the
  ray's path (its start and end, the means of its bounce points, its
  uppermost/lowermost/rightmost/leftmost bounces overall and among only its
  self-bounces) plus two fixed ones (the origin and the point (-1,1)).

  A coloration is a color model (RGB or HSV) plus three channels. Each channel has a
  source:
    Query     a metric, evaluated once per ray for the pixel it was fired
              from - either a point comparison (with its two sample points)
              or one of the other metrics,
    Buddha    named after the Buddhabrot: characteristic points of every ray
              are scattered into a visit-count histogram, and a pixel's
              brightness comes from how often its own location was landed on
              by any ray. Any number of point kinds can be scattered at once
              (means, termination points by reason, every self-bounce, every
              segment midpoint, every point of the path, progressive means),
              and a channel can be limited to the paths of the kinds it ticks:
              escaping, non-escaping, terminated by an obstacle. Counts are compressed to
              0..1 (arctangent or modulus) with an optional brightness boost,
              and the Buddha blur tab spreads each scattered point over a
              neighborhood (Gaussian, triangular, bokeh or catadioptric
              bokeh; radius 0 means no blur; a change recolors the image
              at once, no new Start needed),
    Constant  always 0 or always 1, taking no part in the simulation.
  Per-pixel results are stored, so changing a coloration's colour model or channels recolors the
  image without regenerating it (Buddha channels rebuild their histograms
  first; that never restarts the generation either).

  Several colorations can exist at once, listed in the Colorations tab (of the
  Perception tab). Whichever one is selected is the one shown in the preview
  (with its name overlaid, in the preview only), and the one an export
  includes unless "Include all colorations" is on: then exports and videos
  include all of them, side by side, so more than three channels' worth of
  information can be shown at once.

Framing
  A framing is the camera: where it is (x, y: the middle of what it sees), how
  far across it sees (its diagonal) and the shape of that (its aspect ratio,
  width / height). Framings are listed in the Framing tab (of the Perception
  tab), the same way colorations are; whichever is selected is the one the image
  on screen and every export are of. The default one, diagonal 4 sqrt 2 and
  aspect ratio 1, sees the whole world, -2 to 2 both ways. The image on screen
  takes the shape of the aspect ratio; selecting a framing, or editing the
  selected one, generates the image again - of the same figure, without a new
  Start. A framing is not part of the figure, and its four numbers can be animated
  by keyframes (the camera pans, zooms and changes shape between them; a video
  has the framing of each frame). Buddha seeding with the camera's
  pixels seeds from the pixels of the framing's image. A coloration's picture in
  the list is of framing 0 and a framing's is coloured by coloration 0.

Decorations
  The outline of each obstacle is drawn into the image and so appears in
  exports. An outline has two layers, each about 1.5 pixels thick: the outer
  one shows the obstacle's incoming ray effect and the inner one its outgoing
  effect (pale blue for Reflect, brown for Terminate, black for Pass through).
  Everything else (the dashed outline of not-yet-generated shape
  changes, the previewed ray's path, markers and guide lines, the color
  sample points) is drawn over the preview only and never saved.

Timeline and video
  A keyframe has a frame index and a set of settings - any subset of every setting in the
  Simulation, Fuzzing, Buddha blur and Decorations tabs plus every
  obstacle's and every coloration's own fields (an obstacle's shape parameters and
  ray effects included; its kind is not a setting). "New keyframe (from current settings)" defines every setting
  from the controls' current values; "New keyframe (empty)" defines none.
  Selecting a keyframe loads into the controls every setting it defines and
  every setting its neighbors animate through it (settings nobody defines are
  left alone). While a keyframe is selected, any setting you change gets a blue
  outline on its control and in the keyframe's settings list, until you save it
  ("Save all modified", or a setting's own Save) or deselect the keyframe (click
  the empty part of the keyframe list, or select another one) - the values then
  stay in the controls, but no longer count as modified, so selecting another
  keyframe can overwrite them. Each setting can also be deleted from the keyframe
  individually (or all at once), so different settings can start and stop
  animating at different points on the timeline. Changing the controls never
  touches a keyframe until you save. "Split"
  makes a duplicate right after the original, with the same frame index;
  editing the selected keyframe's frame-index field re-sorts the list. Every
  keyframe always has a frame index (0 for the first one created, one more
  than the highest existing index for each new one after that).

  Each setting is resolved independently, using only the keyframes that
  define it. Between two of them, float-slider settings (shape dimensions and
  rotations, fuzzing angles/jitters/multipliers, segment multiplier, blur
  radius, obstacle geometry and position, and so on) are interpolated
  linearly; everything else (which kind of
  shape, which metric, colors, and so on) holds the earlier keyframe's value
  until the later one is reached, since it isn't a number that could be
  blended.

  A render walks every whole frame from the lowest to the highest frame index
  (optionally skipping the first), draws every frame at the chosen width and
  height (the same world area as the preview), and writes each as a PNG into
  exports/TEMP, next to this program. It then combines them with ffmpeg into
  an .mp4 (h264), .gif, .apng or .mkv (ffv1), or, through the external cjxl
  tool, an animated .jxl (lossless or lossy), saved in the exports folder; the bitrate mode
  (constant quality, target or variable bitrate) and a downsampling factor
  apply to that combined video only, never to the PNG frames. "Re-encode"
  combines whatever frames are already in TEMP using the current encoding
  settings. A render refuses to start if two keyframes share a frame index or
  if the TEMP folder already has any files in it. This program never deletes a
  file on its own; the one exception is the Video panel's Clear button, which,
  after a confirmation, deletes only frame_NNNNN.png files from TEMP.

Exporting a still image
  Export designer: whether the export includes all colorations or only the selected
  one, where to put the coloration description (none, overlaid on the image,
  or in a bottom margin), where to put the figure settings (none, or a left
  margin listing every setting that defines the figure: the termination limits,
  segment length, fuzzing and every obstacle's own fields; the margin grows
  extra columns if they need the room) and where to put the framing settings
  (none, or the same left margin, below the figure's with a line of dashes
  between them); these apply to videos as well, with the values of that frame. Export: either the
  Current image (the preview as it was generated) or a New image generated
  afresh at its own width and height (per coloration), saved as jxl (the default),
  png or qoi (jxl lossless, or lossy with a quality setting). "Export" saves into the
  exports folder (next to the program) under a default name; "Export As..." asks for one. Both are
  disabled until an image has finished generating.

Window layout (six panels, in three columns)
  top left       the generated image (up to 512x512, in the shape of the framing's
                 aspect ratio). Hover to preview a ray; click
                 to pin the preview to that pixel, freeing the mouse.
  top middle     Controls, in eight tabs: Emissions, Termination, Mutations,
                 Perception (Framing, Colorations), Buddha (Buddha seeding, Buddha
                 blur), Decorations, Obstacles and Simulation (empty for now);
                 Start and Revert settings are always visible
  top right      Timeline: the keyframe list, each
                 keyframe's settings, and the selected keyframe's controls
  bottom left    Sequence preview: the previewed ray, in Numbers, Bar graph and Corkscrew tabs
  bottom middle  the Export designer, and below it two tabs: Image (what to
                 export, its format, Export / Export As) and Video (size,
                 starting frame, fps, format, downsampling, bitrate, Start Render /
                 Re-encode / Clear with a progress bar, a preview of the latest
                 frame and a status box; its long explanation is behind a "?"
                 button)
  bottom right   empty for now

Projects
  File > New / Open / Save / Save as (Ctrl+N, O, S, Shift+S) keep a whole setup in a
  .lightbounce file: the simulation settings, decorations, every obstacle, coloration and
  keyframe, and the Export designer and video choices (not the generated image: after
  opening a project the image is blank and Start makes it again, identical because
  fuzzing is seeded). A project file is pretty-printed, versioned JSON, so it can be read,
  diffed and kept in version control. The window title shows the project's name and a *
  while there are unsaved changes; New, Open and closing the window ask first. The dialogs
  start in the projects folder next to the program.

Saved image files (all written to the exports folder) are named
  <program name>-YYYYMMDD-hhmmss-<simulation settings>.<png|qoi|jxl>
and rendered videos
  <program name>-video-YYYYMMDD-hhmmss.<mp4|gif|apng|jxl>  (.ffv1.mkv for mkv)
A JPEG XL file (still or video) also says its quality right before the extension:
name.q90.jxl for lossy quality 90 (5 to 95), name.q100.jxl for lossless.
where the program name is taken from this file's own name (so it follows any
renaming).

Requirements:  pip install numpy numba ffmpeg-python   (tkinter ships with
Python). The ffmpeg command-line tool must also be installed and on PATH for
video rendering and for saving a still as JPEG XL; the cjxl command-line tool
(part of libjxl) is additionally needed for animated JXL video. All three are
otherwise unnecessary: PNG and QOI stills need only the packages above.
Pillow is optional too, and used only for the development screenshots.

Development aids (screenshots and dumps)
  If a dev folder exists next to this program, then once per launch (shortly
  after the numba warm-up finishes) the program grabs a picture of its own
  window into data/debug/screenshots/<file version>-YYYYMMDD-hhmmss-startup.png and
  then pickles everything it holds (except large arrays) into
  data/debug/dumps/<file version>-YYYYMMDD-hhmmss-startup.pkl. The Debug menu does
  the same on demand (suffix -manual; a dump with or without large arrays),
  and an unhandled error leaves a dump without large arrays (suffix -crash) but
  no screenshot. The dumping code is dev/debug_dump.py, which also reads a dump
  back (load_dump). Nothing happens if the dev folder is missing (or Pillow,
  for screenshots).

License
  Copyright (C) 2026 John Dorsey. This program is free software: you can
  redistribute it and/or modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version. It is distributed in the
  hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
  the file LICENSE. The third-party software it uses is listed in
  THIRD_PARTY_NOTICES.md.
"""
from . import app_buddha, app_coloration, app_controls, app_debug, app_engine, app_export, app_framing, app_generation, app_layout, app_obstacles, app_preview, app_projects, app_recolor, app_render, app_timeline, app_video, constants, core, debug, engines, export_text, framings, external, figure, geometry, image_io, kernels, metrics, buddha, color_pipeline, live_image, parsing, stored_layout, paths, projects, selfchecks, settings, settings_model, shapes, ui_text, version, colorations, widgets, window

# Globals that tests patch, and the module that owns each. Code reads them as `paths.EXPORT_DIR` (an attribute of
# the owner) at call time, never by `from .paths import EXPORT_DIR`, which would copy the value at import and make a
# later patch do nothing. They are deliberately NOT re-exported below: `PrismaticFractals.EXPORT_DIR` would be a stale
# copy after a patch, so it does not exist - use `PrismaticFractals.paths.EXPORT_DIR`.
PATCHABLE_GLOBALS = {"EXPORT_DIR": "paths", "TEMP_DIR": "paths", "PROJECT_DIR": "paths", "DEV_DIR": "paths", "DEBUG_DIR": "paths",
                     "PROGRAM_NAME": "paths", "ffmpeg": "external", "ImageGrab": "external",
                     "dump_app_state": "debug"}

_MODULES = (version, paths, external, image_io, widgets, kernels, constants, settings, geometry, shapes, stored_layout, figure, metrics, buddha, colorations, engines, color_pipeline, framings, live_image, export_text, projects, selfchecks, parsing, settings_model, window, ui_text, debug, app_projects, app_export, app_framing, app_render, app_video, app_timeline, app_debug, app_engine, app_generation, app_preview, app_recolor, app_buddha, app_obstacles, app_coloration, app_controls, app_layout, core)

# Everything every module defines is also reachable as an attribute of the package (public and underscore
# names alike), so `PrismaticFractals.write_png` works wherever it lives; tests read names this way. To PATCH a
# global, patch the module that owns it (`PrismaticFractals.paths.EXPORT_DIR`): functions read their own module's
# globals.
for _module in _MODULES:
    for _name, _value in vars(_module).items():
        if not _name.startswith("__") and _name not in PATCHABLE_GLOBALS:
            globals()[_name] = _value
del _module, _name, _value

# The consistency checks between the tables of the modules (see selfchecks.py): once everything is loaded.
selfchecks.run_self_checks()
