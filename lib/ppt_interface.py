import win32com.client
import os
import random
import numpy as np
import PIL
from tqdm import tqdm
import random 




class PPT_shapes():
    def __init__(self):
        self.shapes = { 
            'Title': (None, 'title'), # uses AddTitle
            'TextBox': (None, 'textBox'), # uses AddTextBox
            'StraightConnector': (None, 'line'), # uses AddLine
            'Rectangle': (1, 'shape'),
            'LeftArrow': (34, 'shape'),
            'Chevron': (52, 'shape'),
            'Oval': (9, 'shape'),
            'Donut': (18, 'shape'),
            'IsoscelesTriangle': (7, 'shape'),
            }
        
        self.classIDs = {k:i for i, (k,v) in enumerate(self.shapes.items())}

    def read(self, shape):
        name = "".join(shape.Name.split(' ')[:-1])
        obj_type = self.shapes[name][1]
        if obj_type in ['title', 'textBox', 'shape']:
            func = read_shape
        elif obj_type in ['line']:
            func = read_line
        else:
            raise NotImplementedError

        info =  func(shape)
        info['name'] = name
        info['class'] = self.classIDs[name]
        return info
        


def RGB(rgb):
    return rgb[2]*256**2 + rgb[1]*256 + rgb[0]

def add_shape(slide, ID, x,y,w,h, rgb, border=0, fill = 1, text=0):
    '''
    rgb must be 0-1 ranges
    text just signifies if it has text, that one is a bit of a text to see how good the textures we can
    pick up are
    pages are 720 x 540 - IMO lets enter objects as % width (so that they are 0-1 range)
    
    '''
    # this is writing the properties
    shape = slide.Shapes.AddShape(ID, x,y,w,h)
    # to read them, 
    shape.Fill.Solid() # gradient being the default must be a callback from some weird past
    shape.Fill.ForeColor.RGB  = RGB(np.array(rgb)*255)
    shape.Fill.Visible = fill
    shape.Line.Visible = border
    
    if text > 0: # eventually we'll be able to make a rule for whether white or black text is better 
        # based on fill rgb
        shape.TextFrame.TextRange.Text = 'xx'
        shape.TextFrame.TextRange.Font.Color.RGB  = RGB([0,0,0])
        shape.TextFrame.TextRange.Font.Size = 18

    return shape

def add_line(slide, x1,y1, x2,y2, rgb = [0,0,0]):
    '''
    https://docs.microsoft.com/en-us/office/vba/api/powerpoint.shape.line
    '''
    line = slide.Shapes.AddLine(x1,y1,x2,y2).Line
    line.ForeColor.RGB = RGB(np.array(rgb)*255)
    return line

def add_title(slide,text="Title"):
    '''
    https://docs.microsoft.com/en-us/office/vba/api/powerpoint.shapes.addtextbox
    '''
    
    title = slide.Shapes.AddTitle()
    title.TextFrame.TextRange.Text = text
    return title


def add_textBox(slide,  x,y,w,h, textOri = 1, text = "xxxxxxxx \n \n xxxxxxxx \n \n xxxxxxxx"):
    '''
    https://docs.microsoft.com/en-us/office/vba/api/powerpoint.shapes.addtextbox
    '''

    tb = slide.Shapes.AddTextBox(1, x,y,w,h)
    tb.TextFrame.TextRange.Text = text
    return tb
    
def long_to_rgb(C):
    R = C % 256
    G = C // 256 % 256
    B = C // 65536 % 256
    
    return np.array([R,G,B])/255


def read_shape(shape):
    '''
    https://docs.microsoft.com/en-us/office/vba/api/powerpoint.shape
    '''
    x=shape.Left
    y=shape.Top
    height = shape.Height
    width = shape.Width
    
    fillRGB =long_to_rgb(shape.Fill.ForeColor.RGB)
    fill = abs(shape.Fill.Visible) # sometimes returns -1 if not set, but its still visible in that case
    border = abs(shape.Line.Visible)
    if shape.TextFrame.TextRange.Text != '':
        text = 1
    else: # just detect the presence of text atm, to make it easier to fill in.
        text = 0
    return {
            'left': x,
            'top': y,
            'height': height,
            'width': width, 
            'fillRGB': fillRGB, 
            'border': border,
            'fill': fill,
            'text': text,
            }

def read_line(line, bb_margin=3):
    '''
    https://docs.microsoft.com/en-us/office/vba/api/powerpoint.lineformat
    '''
    left, top, height, width = line.left, line.top, line.height, line.width
    fillRGB =long_to_rgb(line.Line.ForeColor.RGB)
    if height == 0: # horizontal line
        return {'left': left, 'top': top-bb_margin, 'height': bb_margin, 'width': width, 'color': fillRGB}
    elif width == 0: # vertical line
        return {'left': left-bb_margin, 'top': top, 'height': height, 'width': bb_margin, 'color': fillRGB}
    else:
        print("Currently only handling horizontal and vertical lines")
        raise NotImplementedError


def convert_line_xyhw_to_points(x,y,w,h, bb_margin=3):

    if h == bb_margin: # horizontal
        x1 = x
        y1 = y+3
        x2 = x + w
        y2 = y1 
    else: # vertical
        x1 = x+bb_margin
        y1 = y
        x2 = x1
        y2 = y1+h 

    return  x1,y1,x2,y2


def read_slide(slide, shape_manager):
    readout = []
    for shape in slide.Shapes:
        readout.append(shape_manager.read(shape))
    return readout

def write_slide(readout, Presentation, shape_manager, slide=None):
    if slide == None:
        slide = Presentation.Slides.Add(1, 11)
        slide.Shapes.Title.delete()

        for o in readout:
            info = shape_manager.shapes[o['name']]
            if info[1] == 'title':
                add_title(slide)
            elif info[1] == 'textBox':
                add_textBox(slide, o['left'], o['top'], o['width'], o['height'])
            elif info[1] == "shape":
                add_shape(slide, info[0], o['left'], o['top'], o['width'], o['height'], rgb=o['fillRGB'])
            elif info[1] == "line":
                # this one is a little different becasue we write as x1,y1, x2,y2 but read as x,y,h,w with a lil margin. So! Get the x1,y1 as the 
                x1,y1,x2,y2 =  convert_line_xyhw_to_points(o['left'],o['top'],o['width'],o['height'])
                add_line(slide, x1,y1,x2,y2)