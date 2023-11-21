import taichi as ti
import pygame as pg 
import numpy as np 
from pygame import gfxdraw as gfx
import math
pg.init()
ti.init(arch=ti.gpu)

n = 800
pixels = ti.field(dtype=float, shape=(n , n , 3))
class Slider : 
    def __init__(self , x , y , w , h , val , screen) : 
        self.x = x
        self.y = y 
        self.w = w 
        self.h = h 
        self.screen = screen 
        self.val = val/100
        self.thick = 20
        self.col_bar = 5,255,0,100
        self.col_rect = 255,255,255,100
        # self.show()
    def showSlider(self) : 
        button = pg.mouse.get_pressed()
        if button[0] != 0 : 
            pos = pg.mouse.get_pos()
            xPos = pos[0]
            yPos = pos[1]
            if xPos > self.x and xPos < self.x + self.w and yPos > self.y and yPos < self.y + self.h: 
                self.val = (xPos - self.x) / self.w
        self.show()
    def show(self) : 
        gfx.filled_polygon(self.screen, ((self.x + self.val*self.w,self.y), (self.x + self.val*self.w + self.thick,self.y),(self.x + self.val*self.w + self.thick,self.y+self.h),(self.x + self.val*self.w,self.y+self.h)), self.col_bar)
        gfx.polygon(self.screen, ((self.x,self.y), (self.x+self.w + self.thick , self.y),(self.x+self.w + self.thick , self.y+self.h),(self.x , self.y+self.h)), self.col_rect)
@ti.func
def scale(val , startX , endX , startY , endY) : 
    x1 , y1 = startX , startY
    x2 , y2 = endX , endY
    return y2 - (y2-y1)*(x2-val)/(x2-x1)
def scalep(val , startX , endX , startY , endY) : 
    x1 , y1 = startX , startY
    x2 , y2 = endX , endY
    return y2 - (y2-y1)*(x2-val)/(x2-x1)
@ti.func
def color(value , range) : 
    hue = ti.min(255, scale(value , 0 , range , 0,255))
    r = hue
    b = 255 - hue
    g = 255 - abs(r-b)
    return ti.Vector([r , g , b])
@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])
@ti.kernel
def paint(zoom : float , dx:float , dy:float , num_iter : int):
    max_iter = num_iter
    x_shift  = dx 
    y_shift = dy 
    for i, j , k in pixels:  # Parallelized over all pixels
        a = scale(i , 0 , n , -zoom-x_shift ,zoom-x_shift )
        b = scale(j , 0 , n , -zoom-y_shift ,zoom-y_shift )
        ca = a
        cb = b
        iter = 0 
        while iter < max_iter  : 
            aa = a*a - b*b
            bb = 2 * a * b 
            a = aa + ca 
            b = bb + cb
            if abs(a+b) > 4 : break 
            # z = complex_sqr(z) + c 
            iter += 1 
        col = scale(iter , 0 , max_iter , 0 , 1)
        col = scale(col**.5 , 0 , 1 , 0 , 255)
        # if iter == max_iter : col = 0 
        pixels[i, j, 0] = col

        # if iter == max_iter : newColor = ti.Vector([0,0,0])
        # newColor = color(iter , max_iter)
        # pixels[i, j, 0] = newColor[0]
        # pixels[i, j, 1] = newColor[1]
        # pixels[i, j, 2] = newColor[2]

def main() : 
    width = n
    height = n
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption('Title')
    clock = pg.time.Clock() 
    running  = True
    t = 0.0
    painting = pixels.to_numpy()
    painting += 255
    zoom  = Slider(2,15,200,20 , 0 , screen)
    iterations  = Slider(100,15,500,20 , 15 , screen)
    pmx , pmy = pg.mouse.get_pos()
    mx , my = pg.mouse.get_pos()
    shiftX , shiftY = 0.368187500000003 , 0.6492124999999994
    magnify = .049999999999998365
    dmag = .01
    max_iter = 100
    while running : 
        screen.fill((0,0,0))
        max_iter =  int(10+iterations.val*1500)
        # max_iter = max(100 , min(1000 , int(1 / magnify)))
        # max_iter = int(scalep(magnify , .05 , 6.0085317130580144e-06 , 100 , 1200)) 
        # print(max_iter1)
        paint(magnify , shiftX , shiftY , max_iter)
        if pg.mouse.get_pressed()[0] : 
            mx , my = pg.mouse.get_pos()
            if my > 40 :
                shiftX += (mx - pmx)/n * magnify
                shiftY += (my - pmy)/n * magnify
        pmx , pmy = pg.mouse.get_pos()
        painting = pixels.to_numpy()
        pg.surfarray.blit_array(screen , painting)
        iterations.showSlider()
        dmag = .05 * magnify
        keys=pg.key.get_pressed()
        if keys[pg.K_w]:
            magnify += dmag
        if keys[pg.K_e]:
            magnify -= dmag
        for event in pg.event.get() : 
            if event.type == pg.QUIT : 
                running = False 
            elif event.type == pg.KEYDOWN : 
                if event.key == pg.K_ESCAPE : 
                    running = False 
                if event.key == pg.K_r : 
                    print(magnify , shiftX , shiftY , max_iter) 
        pg.display.flip()
        clock.tick(60)
        pg.display.set_caption(f'(press w or e to zoom) , FPS : {round(clock.get_fps() , 2)}')

    pg.quit()
    
if __name__ == "__main__":
    main()
