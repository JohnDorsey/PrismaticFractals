"""What an image on screen was generated from. Pressing Start fixes a figure: the ray engine's settings, the natural
termination settings, and the obstacles as they were, together with the arguments the kernels take that are made from
them. One frozen GeneratedFigure holds all of it, so it changes atomically and a reader never sees half of one figure
and half of another. Before the first Start (and after New or Open) there is none: App.generated is None, and the
PLACEHOLDER_FIGURE (the defaults, no obstacle) stands in where something must be read."""
from dataclasses import dataclass
from .engines import layout_of_engine, tables_of_engine
from .kernels import NaturalTerminationRule, PathArgs
from .settings import (
    DEFAULT_ENGINE_SETTINGS, DEFAULT_NATURAL_TERMINATION_SETTINGS, DEFAULT_RAY_SETTINGS, EngineSettings,
    NaturalTerminationSettings, RaySettings, path_args, settings_natural_termination, settings_ray_args
)
from .shapes import build_obstacle_arrays, obstacles_signature
from .stored_layout import StoredLayout


@dataclass(frozen=True, eq=False)
class GeneratedFigure:
    ray: RaySettings                           # the ray engine's settings it was started with
    natural_termination: NaturalTerminationSettings
    engine: EngineSettings                     # which path engine made it
    termination: NaturalTerminationRule        # the kernels' form of natural_termination
    engine_args: PathArgs                      # the kernels' form of the engine, its settings and the obstacles as generated
    layout: StoredLayout                       # everything its engine stores per path (what the live image keeps)
    obstacle_snapshots: tuple                  # frozen copies of the obstacles, for the export's figure settings text
    obstacles_signature: tuple                 # their plain-data states, to see whether the scene changed since
    obstacle_states: dict                      # obstacle id -> state, for the dashed "changed since" guide

    @property
    def rng_seed(self):
        """The fuzzing seed the kernels take alongside `termination` and `engine_args`."""
        return self.ray.fuzz.rng_seed


def generated_figure(ray, natural_termination, obstacles, engine=DEFAULT_ENGINE_SETTINGS):
    """The GeneratedFigure of Start: the settings records (the engine is the only one there is, for now) and the live
    `obstacles` as they are now."""
    return GeneratedFigure(
        ray=ray, natural_termination=natural_termination, engine=engine,
        termination=settings_natural_termination(natural_termination),
        engine_args=path_args(engine.engine, settings_ray_args(ray), build_obstacle_arrays(obstacles),
                              tables=tables_of_engine(engine.engine)),
        layout=layout_of_engine(engine.engine),
        obstacle_snapshots=tuple(o.snapshot() for o in obstacles),
        obstacles_signature=obstacles_signature(obstacles),
        obstacle_states={o.id: o.state() for o in obstacles})


PLACEHOLDER_FIGURE = generated_figure(DEFAULT_RAY_SETTINGS, DEFAULT_NATURAL_TERMINATION_SETTINGS, [])


def settings_of(figure):
    """(ray, natural_termination) of a GeneratedFigure, or (None, None) for None (nothing generated yet)."""
    return (None, None) if figure is None else (figure.ray, figure.natural_termination)
