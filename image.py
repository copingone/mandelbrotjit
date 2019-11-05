import numpy as np
import imageio
import numba
from numba import jit
from matplotlib import pyplot as plt
from matplotlib import colors

n = 0

@jit
def mandelbrot(c,maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return 0


@jit(parallel=True)
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j],maxiter)
    return (r1,r2,n3)


def mandelbrot_image(xmin,xmax,ymin,ymax,width=3,height=3,maxiter=80,cmap='hot'):
    global n
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)

    fig, ax = plt.subplots(figsize=(width, height),dpi=72)
    ticks = np.arange(0,img_width,3*dpi)
    x_ticks = xmin + (xmax-xmin)*ticks/img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax-ymin)*ticks/img_width
    plt.yticks(ticks, y_ticks)

    norm = colors.PowerNorm(0.3)
    ax.imshow(z.T,cmap=cmap,origin='lower',norm=norm)
    plt.savefig('./images/{}.png'.format(n))
    n=n+1


def createGif():
    global n
    with imageio.get_writer('./result.gif', mode='I') as writer:
        for filename in range(0,n):
            print(filename)
            image = imageio.imread('./images/{}.png'.format(filename))
            writer.append_data(image)


def main():
    #pool = multiprocessing.Pool()#processes=1)              # start 4 worker processes
    print("Rendering")
    for o in range(10,0,-1):
        d = o
        print(o)
        multiplier = 0.0000001*o
        mandelbrot_image(-0.74877+multiplier,-0.74872-multiplier,0.065053+multiplier,0.065103-multiplier,18,18,4096)
    createGif()


if __name__ == '__main__':
    main()
    