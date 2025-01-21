import yaml

data = { 'train' : 'C:/Users/iialab/Desktop/o2o/v6/train/images/',
         'val' : 'C:/Users/iialab/Desktop/o2o/v6/valid/images/',
         'test' : 'C:/Users/iialab/Desktop/o2o/v6/test/images',
         'names' : ['product'],
         'nc' : 1 }

with open('o2o.yaml', 'w') as f:  
  yaml.dump(data, f)


with open('o2o.yaml', 'r') as f:  
  aquarium_yaml = yaml.safe_load(f)
  print(aquarium_yaml)