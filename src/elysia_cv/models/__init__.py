"""Model plugin package.

Each sub-directory under ``models/`` represents a specific model
implementation and MUST expose a ``plugin.py`` with a ``MODEL_PLUGIN``
variable (or ``get_model_plugin()`` factory).

Layout::

    models/
    ├── classification/
    │   ├── resnet/
    │   │   ├── plugin.py         # MODEL_PLUGIN = ResNetPlugin(...)
    │   │   ├── runtime.py        # build_model / forward_train / forward_test
    │   │   └── native_source/    # free-form model code
    │   └── ...
    ├── detection/
    │   ├── faster_rcnn/
    │   └── community/
    │       └── yolo/
    └── ...
"""