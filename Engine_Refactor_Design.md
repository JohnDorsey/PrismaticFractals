# Engine refactor design: decisions

This document records only the decisions that have been made explicitly about the path-engine refactor (making the
ray-bouncing engine one of several path engines, the Mandelbrot set being the next). Proposals, options and open
questions are not recorded here until a decision is made. Before the refactor is started this document is to be
reviewed for inconsistencies and hidden difficulties.

## D1. Obstacles are not part of any one path engine

Obstacles are not owned by the ray-bouncing engine (or by any other engine). They are editable whichever engine is in
use, and they provide shapes to the engine, which may use them in different ways. For example, with a Mandelbrot
engine the incoming and the outgoing ray effect of an obstacle could be terminate or pass-through, and the reflect
option would either be disabled or do something different.

Stated as the direction being leaned towards, to be kept in mind in every future discussion of the engine refactor.

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
