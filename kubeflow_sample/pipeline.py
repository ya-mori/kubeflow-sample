import os

import kfp


if __name__ == '__main__':
    client = kfp.Client(host=os.getenv("KF_HOST"))
    pipeline_list = client.list_pipelines()
    print(pipeline_list)
