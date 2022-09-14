# Download Models:
# 1, pretrained model
DOWNLOAD=$1

BLOB='https://hdvila.blob.core.windows.net/dataset/pretrained.zip?sp=r&st=2022-09-13T08:25:54Z&se=2024-12-31T16:25:54Z&spr=https&sv=2021-06-08&sr=b&sig=Zt8vmQ%2F5wU35507Dar4i4Qsk3dqf15aEBQOS4QqUUrc%3D'

# 1, pretrained model
wget -nc $BLOB -O $DOWNLOAD/pretrained.zip
unzip $DOWNLOAD/pretrained.zip -d $DOWNLOAD

BLOB='https://hdvila.blob.core.windows.net/dataset/data.zip?sp=r&st=2022-09-13T02:35:13Z&se=2024-12-31T10:35:13Z&spr=https&sv=2021-06-08&sr=b&sig=BjQXSegSvllCLpx%2B%2FhDH6VwVDE0e2XHZ%2FqwAo5ZpyeQ%3D'

# 2, downstream dataset
wget -nc $BLOB -O $DOWNLOAD/data.zip
unzip $DOWNLOAD/data.zip -d $DOWNLOAD
mv $DOWNLOAD/downstream_data $DOWNLOAD/data
