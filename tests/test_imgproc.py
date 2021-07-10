from src.data import imgproc

def test_transform():
    path = "./tests/view1_frontal.jpg"
    proc_class = imgproc.get_proc_class('skimage')
    image = proc_class.imread(path)
    transformations = [
        ('resize', {'size': (320, 320)}),
        ('flatten', {})
    ]
    result = proc_class.transform(image, transformations)
    print(result)
