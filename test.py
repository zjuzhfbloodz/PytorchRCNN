from imVisual import readImage, readImageTensor, random_color, show, plot_image, names, colors
import torchvision
from torchvision.io import read_image


def test_image(model,path,img_size,threshold=0.9,save_name='test.jpg',width=1,font_size=50):
    # 3 type of image reading
    image = readImage(path,size=img_size)
    image2 = read_image(path)
    image3 = readImageTensor(path,size=img_size)
    x = image.float()
    
    model.eval()
    out = model([x])[0]
    
    mask = out['scores']>threshold
    boxes = out['boxes'][mask]
    labels = out['labels'][mask]
    scores = out['scores'][mask]
    
    plot_image(image,boxes,labels,names,colors,save_name=save_name,width=width,font_size=font_size)


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    print('model ready')
    path = './data/3.JPG'
    test_image(model, path, img_size=(800,1200), threshold=0.5, save_name='./data/test.jpg')