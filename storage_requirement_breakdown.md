Breakdown of the `data` directory's (where datasets are downloaded) storage requirements:

```
$ du -h --max-depth=1 data
1.2G	data/purchase-100
3.1G	data/celeba
178M	data/cifar-10-batches-py
3.9G	data/texas-100
178M	data/cifar-100-python
481M	data/tiny-imagenet-200
9.4G	data
```

Breakdown of the `experiments` directory's (where experiments are saved) storage requirements:
```
$ du -h --max-depth=1 experiments
34M 	experiments/celeba
16M 	experiments/celeba-old
1.2G	experiments/purchase100
3.3G	experiments/texas100
2.6G	experiments/tiny-imagenet-200
116M	experiments/cifar100
16G 	experiments/cifar10
23G 	experiments
```
