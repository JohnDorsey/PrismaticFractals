# Engine refactor design: decisions

This document records only the decisions that have been made explicitly about the path-engine refactor (making the
ray-bouncing engine one of several path engines, the Mandelbrot set being the next). Proposals, options and open
questions are not recorded here until a decision is made. Before the refactor is started this document is to be
reviewed for inconsistencies and hidden difficulties.

## D1. Obstacles are not part of any one path engine

Obstacles are not owned by the ray-bouncing engine (or by any other engine). They are editable whichever engine is in
use, and they provide shapes to the engine, which may use them in different ways.

## D2. Engine kernels stay in one file (`lightbounces/kernels.py`)

All numba kernels, of every engine, stay in `lightbounces/kernels.py`; engines are sections of that file, with the shape
kernels shared by all of them. The existing numba cache rule and its test stay as they are, and no custom numba cache
locator is introduced.

Revisit this decision if a third engine is added.

## D3. The stored layout is per engine (A2)

Each engine declares the sequences it has, and the per-pixel arrays (the metrics and the characteristic points) are
sized from what that engine stores, rather than every engine using one layout built for the ray-bouncing engine's four
sequences. The reason given is memory.

How this leads to it: `lightbounces.stored_layout.StoredLayout` (FILE_VERSION 151) is the mechanism. A layout names
the metric slots and characteristic point slots an image keeps, and where they sit in its arrays;
`lightbounces.kernels.trace_rows_stored` writes through those slot lists and `lightbounces.color_pipeline` and
`lightbounces.buddha` read through them. It was first used for something else: an export keeps only the slots its
colorations read (`lightbounces.color_pipeline.demand_layout`), which was measured to cut the peak memory of a
3840x2160 export from about 2.9 GB to about 0.9 GB for the default coloration. A per-engine layout is the same
object built from a different source: an engine's own layout (the sequences it has), of which a render keeps the part
its colorations need.

## D4. No tab or control disappears when the engine changes; unusable ones are greyed out

Outside the tab that controls which engine is in use, no tab disappears depending on the engine in use, and no control
disappears depending on the engine in use. A control the engine in use cannot make use of is greyed out instead.

## D5. What the Mandelbrot engine uses

The Mandelbrot path engine has the entire contents of the Emissions tab and of the Mutations tab greyed out. It makes
use of all the settings in all the other tabs.

## D6. The Simulation tab chooses the engine

The Simulation tab holds a dropdown box that chooses between two path engines. Its options are "Ray Bouncing" and
"Mandelbrot". Underneath this control, engine-specific settings may appear, if they do not have a tab of their own
somewhere else.

## D7. Switching engine resets the settings the new engine does not support

The settings the newly selected engine does not support are set to their default values for that engine. When this
happens a message pops up, with this text and layout (one group per tab; a numeric setting shows its old value and its
new value, a choice its old choice and its new choice):

    Some settings that weren't supported by this engine have been reset:
      in the <tab name> tab:
        <setting name>: <old value> -> <new value>
      in the <tab name> tab:
        <setting name>: <old choice> -> <new choice>
    [ok] [go back]

"go back" switches the selected engine back to what it was before and puts every setting that was just modified for
the new engine back the way it was. Which value a setting is reset to is given by D17.

## D8. The engine is figure-defining

The selected engine is figure-defining: a change of engine is a change of the figure (it needs Start).

## D9. Obstacles decide which pixels get a path, in every engine

Obstacles still decide which pixels get a path (their "fire inside" setting), whichever engine is in use, the Mandelbrot
engine included.

## D10. The default world is doubled: (-2, -2) to (2, 2)

The default framing (framing 0), for all engines, is now (-2, -2) to (2, 2) (it was (-1, -1) to (1, 1)). The default
radius of every obstacle, and every other default parameter of an obstacle that is measured as a distance in world
units, is doubled with it, default widths and default heights included. The default Buddha seeding zone is doubled too.
The parameters affected include: the width and height of ellipses and superellipses, the radii and the separation of
lenses, the length and the radius of capsules, the cut radius and the cut offset of lunes, the half-length and the
half-width of eggs, the size of Moss eggs, and the pitch of spirals. Every setting whose default value is doubled also has its maximum value
in the interface doubled, and the maximum values of r, cx and cy of an obstacle's position are doubled as well, and the minimum values of cx and cy become -2. What
happens to the "Top left corner" constant
reference point is in D24, to the metrics' divisors and the default escape radius in D23, and to the snail of
Pythagoras (which has no size control yet) in D25.

## D11. Reflect is disabled for the Mandelbrot engine

For now the Reflect effect is disabled for the Mandelbrot engine. Switching the engine to Mandelbrot automatically
replaces an obstacle's Reflect effect, both for crossing in and for crossing out, with pass-through, with a message to
the user (see D7). If a Mandelbrot render nevertheless runs while an obstacle with an incompatible effect exists, the
paths are terminated by that obstacle, for now.

## D12. A Mandelbrot orbit meets obstacles through the existing termination machinery

The interaction of a Mandelbrot orbit with an obstacle reuses the existing termination machinery. Each segment of the
orbit, z_k to z_(k+1), is tested against the obstacles (not only the points z_k).

## D13. A greyed-out control means the setting is useless for the current engine, nothing more

A greyed-out control only tells the user that changing the setting would be useless, because the current engine is not
affected by it. The setting still exists, still gets saved and loaded in project files, still affects outline
decorations, and can still be animated by keyframes. The figure settings margin text includes the settings that are
irrelevant to the current engine, in medium grey. Settings are to be phased out of file names altogether, so greyed-out
settings are not specifically removed from file names.

## D14. The Mandelbrot engine's own settings

The engine-specific settings that appear under the engine dropdown (D6) for the Mandelbrot engine are: the exponent,
the starting value z0, whether a pixel maps to c (Mandelbrot) or to z0 (Julia), and the Julia constant. All four are
included, and the numeric ones are animated by keyframes.

## D15. The sequence preview and the guide lines stay visible

The sequence preview panel and the decorations stay visible in every engine. The "Termination point guide lines" are
still shown when the Mandelbrot engine is in use, in case a path terminates somewhere within the preview.

## D16. Buddha scatters can be limited by how the paths ended

Every colour channel's settings have a filter that is a set of three checkboxes, one for each kind of path listed
below: the channel scatters only the paths of the kinds that are ticked. The Buddhabrot is in scope.

The three kinds of path are:
- escaping: the path reaches the escape radius;
- non-escaping: the path ended by anything that is neither reaching the escape radius nor an obstacle: the iteration
  limit, the distance limit, confinement, a perpendicular collision;
- terminated by obstacle: the path is terminated by an obstacle. It is neither escaping nor non-escaping.

There is no checkbox for "all the points": the scatter of all the points is had by ticking all three. A Buddhabrot
scatter of a path includes the path's starting point. The existing options "Path: Last point (distance limit)",
"(perpendicular collision)" and "(confined)" stay.

## D17. How an incompatible value is corrected

When a setting's value is not compatible with the engine (D7, D20), it is corrected: a dropdown choice falls back to its
default, a float or an integer is clamped to a value the engine accepts. This holds even where no setting currently
needs it; it is the rule if one is added (for example one engine allowing a negative segment length multiplier and the
other not). The correction stands unless the user presses "go back" in the notice popup (D7).

## D18. Engines remember their settings

When the engine is switched back, the settings the engine had come back, except for the settings the user has changed
since, and the settings that have been changed by selecting a keyframe since. What the engines remember is kept for the
session only: it is not saved in project files.

## D19. The engine and keyframes

The choice of engine can be saved to a keyframe like any other setting. The user gets the popup of D7 when they try to
save a setting value (the choice of engine included) to the selected keyframe in a way that makes the choice of engine
and other settings incompatible, whether within the selected keyframe or with other frames that the selected keyframe
affects. The popup lists the changes it suggests in order to fix the incompatibility, and they are applied if the user
hits [ok]; [go back] cancels the save. They are changes to whichever keyframe settings the user is not currently trying to save or, if the user is
trying to save several incompatible settings at the same time, to the settings other than the choice of engine being
saved. Switching the engine (D7) does not change the contents of any
keyframe: keyframes holding values that are not valid are allowed to exist, and they cause corrected values to be used
for rendering (D20).

## D20. Rendering and opening with incompatible values

If incompatible values are met while a video is exported (for example because a project file was edited by hand), then
for each frame, after the keyframes have applied their values to the settings and before the frame is rendered, and
before its margin text is made, the settings are corrected (D17), and an error message is printed to the status box.

When a project is opened and invalid values have to be corrected, the user is warned anyway, with the message box of
D36 (worded "some values from this project have been clamped or defaulted:", showing the values; its only button is
[ok], so there is no "go back"). A project opened with such corrections counts as having unsaved
changes.

## D21. The status box is only cleared when a new export starts

The status box (the text of the Video panel's render status) only clears its output when a new export is started.
(Until now every message replaced the previous one.) A re-encode counts as a new export.

The frame count "n/total" is to be shown as text on top of the progress bar, for example by one of the methods
described at https://stackoverflow.com/questions/74895125/how-to-customize-tkinter-ttk-progressbar-with-label-text-and-custom-bar-color
If that result cannot be had easily, one line is added to the status box for every frame instead.

Errors of a still-image export appear as popup dialog box messages.

## D22. A greyed-out setting still colours solid obstacle outlines

A setting that is greyed out for the current engine still affects the colours of solid-line obstacle outlines. No case is
known in which a setting affecting a dashed obstacle outline depends on the engine; if one comes up, this decision is to
be asked about again.

## D23. The metrics and the escape radius follow the doubled world

- The divisors of the metrics are scaled with the world: the point-distance metrics divide by 4 where they divided by
  2 (still clipped to 0..1), and the arctangent of a path's length is taken of length / 10 where it was length / 5.
- The default escape radius is 4.0 (it was 2.0), for every engine.
- The kernel constants `WORLD_RECT_DEFAULT` and `DEFAULT_PIXEL_WORLD_WIDTH` are updated to the doubled world.

## D24. The fixed top-left reference point

In project files the existing "Top left corner" point choice is migrated to a new string key "(-1,1)", which is also how
that choice is shown in the point dropdowns. A new "top left point" choice is stored in project files under the key
"(-2,2)", so that changing what the top left means again later needs no migration.

## D25. The snail of Pythagoras gets a size control

The snail of Pythagoras gets a size control named "radius". A project that does not have a radius saved for a snail
loads it with radius 1, the size it has today. A new snail's default radius is 2 (the size it has today, doubled by D10).

## D26. Mandelbrot details

- The starting value z_0 of an orbit is chosen between two choices: 0 and c.
- The exponent can be any real number. A value that is not finite (for example zero raised to a negative power) ends
  the path as an escape: under the clip behaviour of D29 the path ends at its last finite point, tagged as an escape.
  An integer exponent is computed by plain multiplication.
- The Julia constant has two controls, (re, im), for now.
- The Mandelbrot engine comes with tests of its own, such as known iteration counts at known points and an estimate of
  the area of the set.

## D27. Perpendicular collision does not end Mandelbrot paths (for now)

In the Mandelbrot engine a perpendicular collision does not terminate a path, for now. It is not necessary, because
Mandelbrot paths do not bounce off themselves, and the ambiguity of which side of itself a path retracing its own steps
ends up on does not matter there. This may change someday. In the ray-bouncing engine a perpendicular collision
belongs to the non-escaping kind of path of D16.

## D28. One driver per job, with an engine switch inside

The engines share their drivers (the kernels that loop over pixels or seed points and call the engine's path function):
there is one driver for each job (tracing the rows of an image, scattering from a grid of seed points), and each has a
switch on the engine inside, so the gating of pixels, the storing of results and the Buddha scatters (with their
filters) exist once. The alternative of one driver per engine was not chosen. Every caller of an engine's path function,
the hover preview included, goes through the same switch on the engine.

## D29. A control chooses how an escaping Mandelbrot orbit ends

A control chooses between two escape behaviours of a Mandelbrot orbit: the orbit ends where its segment crosses the
escape radius ("clip", as a ray does), or it ends at z_(k+1), the point that went beyond the radius. The default is
clip. The control is in the Simulation tab.

## D30. A path begins if its pixel's location fires

A path begins if the pixel whose world location provides the location variable met the criteria for firing a ray. In the
Mandelbrot case, where the pixel's location becomes c, the path begins if c is inside a shape with "fire rays inside"
checked. In the Julia case, where the pixel's location becomes z_0, the path begins if z_0 is inside such a shape.

When z_0 is 0 (D26) the path's first point is the origin, not c. Which shapes the path is inside is decided at the
path's first point (the origin), while whether the path begins at all is decided at c. When this case is chosen, a note
under the z_0 option in the interface explains this difference between the flagging location and the gating location.

## D31. What the Mandelbrot engine declares

The Mandelbrot engine declares one sequence, "Path" (every point of the orbit, the starting point included), with the
seven point reductions. The metrics it offers are: iteration count, path length, last direction, termination reason
and the four point distances. It does not offer: loop area, bounce angles, last-segment ratio, and the Self, Boundary
and Bounces sequences.

## D32. New settings stay out of file names

The new settings (the engine and its settings) stay out of file names from the start.

## D33. The format 9 migration writes old values explicitly

Wherever a value missing from an older project file would take a default that D10 or D23 changes (for example a missing
framings section, escape radius or confinement threshold, or a missing obstacle value), the format 9 migration writes
the old value into the file explicitly, so that the file keeps its image.

## D34. Choice ids are global across engines

The ids of the choices a coloration channel can make (metrics, points, Buddha options) are global: the union of what all
the engines offer. Each engine offers a subset of them, and has its own stored layout (`StoredLayout`) from an id to a
position in its arrays. Whether a choice is compatible with an engine (D7, D17, D18) is then a test of membership in
that engine's subset.

## D35. One correction function at every render entry

The correction of incompatible values (D17) is one function, applied at every place where a render starts: the live
Start, a still-image export, each frame of a video, thumbnails, and the hover preview. The termination of paths by an
obstacle with an incompatible effect (D11) remains only as a last resort inside the kernel. A new obstacle made while
the Mandelbrot engine is selected gets pass-through for both its effects.

## D36. Selecting a keyframe with invalid values

When a keyframe with invalid values is selected, the values are corrected as they are applied to the current settings
(D17), and not silently: a message box says "some values from this keyframe have been clamped or defaulted:" and shows
the values. Its only button is [ok].

## D37. The bar graph truncates at two units

The bar graph of the sequence preview truncates a segment at two world units (it was one).

## D38. "All path points" is a Buddha option of both engines

"All path points" (every point of the path, the starting point included) exists as a Buddha option for both engines. The
Mandelbrot engine does not offer the options "All self bounces" and "Progressive means of bounce points" (an orbit has no bounces).

Revised at FILE_VERSION 168: the Mandelbrot engine DOES offer "All segment midpoints" and "Progressive means of segment midpoints" (first decided
against). The kernels make both from the path's points alone, they give a smoothed Buddhabrot, and the user's own image showed they work.

## D39. The Mandelbrot engine has defaults of its own

Where the default choice of a setting is not offered by the Mandelbrot engine, the engine's own default is used in its
place (D7, D17). The default coloration's bounce count becomes the Path count, and the default Buddha option "Bounces:
Mean point" becomes "Path: Mean point".

## D40. An export takes what no keyframe defines from the live values when it began

A video, and a new still image, take the value of a setting that no keyframe defines from what the controls held when the
export began, not from the program's default values. (This was already so for obstacles, colorations and framings; it
was not for the other settings, which fell back to the defaults.) The keyframe editor shows the same value.
