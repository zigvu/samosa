##### `fine_tuner.py`

Fine tunes existing caffe model on dataset.

##### `train_config.py`

Sets the config based on `chia/experiments/cfgs/zigvu_end2end.yml` file and input config Hash (from `Rasbari`). This includes zigvu specific config settings in the application.

##### `train_dataset.py`

Data adapter to convert external data source into what `py-faster-rcnn` expects.

##### `train_templater.py`

Changes prototxt template files based on fine tune task.
