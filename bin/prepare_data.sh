echo "Enter your Roboflow private key: "
read private_key

mkdir -p data
wget -O data/dataset.zip https://public.roboflow.com/ds/gfNzKDuCKc?key="$private_key"
unzip -q data/dataset.zip -d data/