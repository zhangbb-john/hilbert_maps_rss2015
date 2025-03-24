# Issue
## "mbkm.fit(data)" error occurs when I run python example.py datasets\intel.gfs.log nystroem
```
PS D:\HKUST_Visiting\record\Research\01 Radar-place-recognition\Code\HilbertSLAM\hilbert_maps_rss2015> python example.py datasets\intel.gfs.log fourier^C
PS D:\HKUST_Visiting\record\Research\01 Radar-place-recognition\Code\HilbertSLAM\hilbert_maps_rss2015> python example.py datasets\intel.gfs.log nystroem
<frozen importlib._bootstrap>:228: RuntimeWarning: scipy._lib.messagestream.MessageStream size changed, may indicate binary incompatibility. Expected 56 from C header, got 64 from PyObject
Traceback (most recent call last):
  File "D:\HKUST_Visiting\record\Research\01 Radar-place-recognition\Code\HilbertSLAM\hilbert_maps_rss2015\example.py", line 215, in <module>
    sys.exit(main())
  File "D:\HKUST_Visiting\record\Research\01 Radar-place-recognition\Code\HilbertSLAM\hilbert_maps_rss2015\example.py", line 196, in main
    model = train_incremental_hm(train_data, args.components, args.gamma, args.feature)       
  File "D:\HKUST_Visiting\record\Research\01 Radar-place-recognition\Code\HilbertSLAM\hilbert_maps_rss2015\example.py", line 86, in train_incremental_hm
    model.fit(np.array(training_data))
  File "D:\HKUST_Visiting\record\Research\01 Radar-place-recognition\Code\HilbertSLAM\hilbert_maps_rss2015\hilbert_map.py", line 87, in fit
    mbkm.fit(data)
  File "C:\Software\Anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py", line 1694, in fit
    self._check_mkl_vcomp(X, self.batch_size)
  File "C:\Software\Anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py", line 874, in _check_mkl_vcomp
    modules = threadpool_info()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 124, in threadpool_info
    return _ThreadpoolInfo(user_api=_ALL_USER_APIS).todicts()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 340, in __init__      
    self._load_modules()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 373, in _load_modules 
    self._find_modules_with_enum_process_module_ex()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 485, in _find_modules_with_enum_process_module_ex
    self._make_module_from_path(filepath)
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 515, in _make_module_from_path
    module = module_class(filepath, prefix, user_api, internal_api)
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 606, in __init__      
    self.version = self.get_version()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 646, in get_version   
    config = get_config().split()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 606, in __init__      
    self.version = self.get_version()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 646, in get_version   
    config = get_config().split()
    self.version = self.get_version()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 646, in get_version   
    config = get_config().split()
  File "C:\Software\Anaconda3\lib\site-packages\threadpoolctl.py", line 646, in get_version   
    config = get_config().split()
    config = get_config().split()
AttributeError: 'NoneType' object has no attribute 'split'
```

* Solution:
```
pip install --upgrade threadpoolctl
```
## 