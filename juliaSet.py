import taichi as ti
import pygame as pg 
import numpy as np 
pg.init()
ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n , 3))

def scale(val , startX , endX , startY , endY) : 
    x1 , y1 = startX , startY
    x2 , y2 = endX , endY
    return y2 - (y2-y1)*(x2-val)/(x2-x1)
@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])
@ti.kernel
def paint(t: float , y : float):
    for i, j , k in pixels:  # Parallelized over all pixels
        # c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        c = ti.Vector([y, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        col = 1 - iterations * 0.02
        # pixels[i, j, k] = 1 - col * k/2
        pixels[i, j, k] = col 

def main() : 
    width = n*2
    height = n
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption('Title')
    clock = pg.time.Clock() 
    running  = True
    t = 0.0
    painting = pixels.to_numpy()
    painting += 255


    while running : 
        screen.fill((0,0,0))
        mx , my = pg.mouse.get_pos()
        x = scale(mx , 0 , n , -1,1)
        y = scale(my , 0 , n , -1,1)
        paint(t,-.8)
        # paint(x,y)
        painting = pixels.to_numpy()
        painting *= 255
        pg.surfarray.blit_array(screen , painting)
        t += .01
        for event in pg.event.get() : 
            if event.type == pg.QUIT : 
                running = False 
            elif event.type == pg.KEYDOWN : 
                if event.key == pg.K_ESCAPE : 
                    running = False 
        pg.display.flip()
        clock.tick(60)
        pg.display.set_caption(f'FPS : {round(clock.get_fps() , 2)}')

    pg.quit()


if __name__ == "__main__":
    main()
