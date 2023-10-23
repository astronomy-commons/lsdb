# LSDB Cloud Tests

### Performing LSDB cloud tests
The only currently implemented cloud platform is abfs. In order to run the tests, you will need to export the following environmental variables in a command line:
```bash
export ABFS_LINCCDATA_ACCOUNT_NAME=lincc_account_name
export ABFS_LINCCDATA_ACCOUNT_KEY=lincc_account_key
```
Then to run the tests:
```bash
pytest cloud_tests/ --timeout 10 --cloud abfs
```
The timeout needs to be increased to account for latency in contacting cloud buckets, and performing heavier i/o commputations. 


### How are we connecting to the cloud resources?

We have abstracted our entire i/o infrastructure to be read through the python [fsspec](https://filesystem-spec.readthedocs.io/en/latest/index.html) library. All that needs to be provided is a valid protocol pathway, and storage options for the cloud interface. 


### Adding tests for a new cloud interface protocol

There are various steps to have tests run on another cloud bucket provider (like s3 or gcs). 

* 1.) You will have to create the container/bucket
* 2.) You will have to edit `cloud_tests/conftest.py` in multiple places:
```python
...
#...line 38...
@pytest.fixture
def example_cloud_path(cloud):
    if cloud == "abfs":
        return "abfs:///hipscat/pytests/"
    
    #your new addition
    elif cloud == "new_protocol":
        return "new_protocol:///path/to/pytest/"

    else:
        raise NotImplementedError("Cloud format not implemented for hipscat tests!")

@pytest.fixture
def example_cloud_storage_options(cloud):
    if cloud == "abfs":
        storage_options = {
            "account_key" : os.environ.get("ABFS_LINCCDATA_ACCOUNT_KEY"),
            "account_name" : os.environ.get("ABFS_LINCCDATA_ACCOUNT_NAME")
        }
        return storage_options
    
    #your new addition
    elif cloud == "new_protocol":
        storage_options = {
            "valid_storage_option_param1" : os.environ.get("NEW_PROTOCOL_PARAM1"),
            "valid_storage_option_param1" : os.environ.get("NEW_PROTOCOL_PARAM2"),
            ...
        }

    return {}
```

* 3.) Finally, you will need to copy the entire `/tests/data/` directory into your newly created bucket. This can be accomplished by running the `copy_data_to_fs.py` script in the `cloud_tests/` directory. 
* 4.) Before running the tests, you will need to export your `valid_storage_option_param` into the environment.