# Download Models and Data:
DOWNLOAD=$1

BLOB='https://hdvila.blob.core.windows.net/dataset/lfvila_release.zip?sp=r&st=2023-03-16T05:01:27Z&se=2027-03-01T13:01:27Z&spr=https&sv=2021-12-02&sr=b&sig=lxR7bZ4i3Jpm4Z93u%2BgqhGvfF6DZ4hyRgPFwhwO9i78%3D'

wget -nc $BLOB -O $DOWNLOAD/lfvila_release.zip
unzip $DOWNLOAD/lfvila_release.zip -d $DOWNLOAD