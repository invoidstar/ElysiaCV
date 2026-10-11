"""Shared reusable components — backbones, necks, heads.

These are the "Lego bricks" that model plugins can optionally compose
to build their architectures.  A plugin is never required to use them;
it can implement its model entirely in ``native_source/``.

All components are registered through ``elysia_cv.core.registry``.
"""