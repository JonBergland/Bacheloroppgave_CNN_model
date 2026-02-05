from CNN import CNN

def main(dataset_root: str, 
         epochs: int = 5,
         lr_rate: float = 0.01,
         momentum: float = 0.09,
         batch_size: int = 32,
         img_size: int = 32, 
         manual_seed: int = 42):

    cnn = CNN(dataset_root=dataset_root, 
    epochs = epochs,
    lr_rate = lr_rate,
    momentum = momentum,
    batch_size = batch_size,
    img_size = img_size,
    manual_seed = manual_seed)

    cnn.train()
    cnn.evaluate()


    
if __name__ == '__main__':
    root = './imagenet-256/sample_classes'
    epochs = 5
    lr_rate = 0.01
    momentum = 0.9
    batch_size = 32
    img_size = 32
    manual_seed = 42


    main(dataset_root=root,
         epochs= epochs,
         lr_rate= lr_rate,
         momentum=momentum,
         batch_size=batch_size,
         img_size=img_size,
         manual_seed=manual_seed)

