import io
import boto3
import botocore
from typing import Union


class S3BucketClient(object):
    def __init__(
        self,
        bucket_name: str,
        region_name: str = str(),
        aws_access_key_id: str = str(),
        aws_secret_access_key: str = str(),
    ) -> None:
        self.__bucket_name = bucket_name
        if (
            aws_access_key_id != str()
            and aws_secret_access_key != str()
            and region_name != str()
        ):
            self.client = boto3.client(
                "s3",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            self.client = boto3.client("s3")

    @property
    def bucket_name(self) -> str:
        return self.__bucket_name

    def check_path(self, path: str) -> bool:
        """
        Returns:
            - False if the path does not exist (404 response code)
            - True if the path exists (200 response code)
        
        Raises an exception if any errors if response code is not 200 or 404
        """
        try:
            self.client.get_object(
                Bucket=self.bucket_name, Key=path,
            )
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                # The object does not exist.
                return False
            else:
                raise Exception(f"")  # TODO
        # The object exists.
        return True


    def get_object(self, path: str) -> Union[bytes, None]:
        """
        Returns:
            _ None if the path doesn't exist
            - Byte value of the stored object in S3 bucked at path
        
        Raises an exception if path does not exist
        """
        check_path = self.check_path(path=path)
        with self.__cv:
            if not check_path:
                # The path does not exist
                raise Exception(f"")  # TODO populate correct error message

            byte_value = bytes()
            with io.BytesIO() as b:
                self.client.download_fileobj(self.bucket_name, path, b)
                byte_value = bytes(b.getvalue())

            return byte_value

    def save_object(self, path: str, byte_value: Union[bytes, None]):
        """
        If byte_value passed is None:
            - Will create a directory path 
        To upload a pytorch model, create a buffer write the model to the buffer and upload it:
            - buffer = io.BytesIO()
            - torch.save(model, buffer)
            - s3bucketClient.upload_byte_object("VOLUME_UUID/MODEL_WEIGHTS.pt", buffer.getvalue())

        If uploading json:
            - byte_value = bytes(json.dumps(a).encode('UTF-8'))
            - s3bucketClient.upload_byte_object("VOLUME_UUID/JSON_DATA.json", byte_value)
        
        Raises an exception if:
            - HTTPStatusCode != 200
            - ResponseMetadata doesn't exist in the response
        """
        try:
            if byte_value != None:
                resp = self.client.put_object(
                    Bucket=self.bucket_name, Key=path, Body=byte_value
                )
            else:
                assert path.endswith("/"), f"path must end with '/', path: {path}"
                resp = self.client.put_object(Bucket=self.bucket_name, Key=path)

            if resp["ResponseMetadata"]["HTTPStatusCode"] == 200:
                return None
            else:
                # HTTPS Status code != 200
                raise Exception(f"")  # TODO 
        except Exception as e:
            # No response meta data
            raise Exception(f"")  # TODO