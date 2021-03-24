
import warnings
import numpy as np
import scipy.interpolate as si
from numpy.matlib import randn
from numpy.polynomial import Chebyshev
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial as pl

import matplotlib.pyplot as plt

from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.io import show
from bokeh.plotting import figure

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.widgets import TextInput, Button, Paragraph

import pandas as pd

import Utils as utils

import logging
logger = logging.getLogger(__name__)


class TestFrame1():

    segs = dict()

    def __init__(self, qlog = None):

        utils.configLogger(logger, qlog)
        logger.info("TestFrame started...")




    def getPolySegment(self, seed, start=0, stop=4, degree=2 ):

        z1 = np.polyfit(np.array(seed[start:stop+1,0]), np.array(seed[start:stop+1,1]), degree, w=seed[start:stop+1,2])
        # z1 = np.polyfit(np.array(seed[start:stop,0]), np.array(seed[start:stop,1]), degree)
        p1 = np.poly1d(z1)
        return p1


    def showPoly (self, poly: np.poly1d, prefix="Poly = "):

        sb = utils.StringBuilder(prefix)
        for coeff in poly.coeffs:
            sb+= ' {:9.6f},'.format(coeff)
        return str(sb)


    def adjustPoints (self, seed, xaxis=(0,1), yaxis=(0,1)):

        outx=(seed[:,0]*xaxis[1])+xaxis[0]
        outy=(seed[:,1]*yaxis[1])+yaxis[0]
        return (outx, outy)


    def scaleAxis(self, seed, axis=(0,1)):
        outx=(seed * axis[1])+axis[0]
        return outx


    def getSegments (self, cv , brks) :

        # cvlg = len(cv)
        segs = list()
        for brk in brks:
            segpoly = self.getPolySegment(cv, brk[0], brk[1], brk[2])
            # chebseg = self.getChebySegment(cv, brk[0], brk[1], brk[2])
            # cfitseg = self.getCFit(cv, brk[0], brk[1], brk[2])
            cfitseg = [0, 0, 0] #self.getCFit(cv, brk[0], brk[1], brk[2])
            seg_start = cv[brk[0],0]
            seg_end = cv[brk[1],0]
            segs.append((segpoly, seg_start, seg_end))

        return segs


    def getCoeffs(self):

        segs = {}

        indexes = ['poly', 'sdx_start', 'sdx_stop', "sdy_start", 'sdy_stop', 'yoffset', 'xoffset', 'x', 'y']

        segs.update( {'buildp' : pd.Series([np.poly1d([27.712956,  1.695464, -0.020245]),
                                            0.0, 0.160000, 0.0, 0.96, 0., 0.,
                                            None, None], indexes)})

        segs.update( {'buildstab' : pd.Series([np.poly1d([-1.756039,  1.093432,  0.830011]),
                                               0.16000, 0.300000, 0.96, 1.0, 0., 0.,
                                               None, None], indexes)})

        segs.update( {'stabjump' : pd.Series([np.poly1d([-20.000000,  7.000000]),
                                              0.30000, 0.320000, 1.0, 0.60, 0., 0.,
                                              None, None], indexes)})

        segs.update( {'stabinit' : pd.Series([np.poly1d([15.625000, -13.125000,  3.200000]),
                                               0.32000, 0.400000, 0.60, 0.45, 0., 0.,
                                               None, None], indexes)})

        segs.update( {'stabend' : pd.Series([np.poly1d([0.151166, -0.140317, -0.241949,  0.559550]),
                                              0.40000, 1.00000, 0.45, 0.328, 0., 0.,
                                              None, None], indexes)})


        # segs.append (np.polynomial.polynomial([-20.000000,  7.000000]), 0.30000, 0.320000)

        # poly2 = np.poly1d([0,0,0])
        # segs.append ((np.polyadd(poly1, poly2), 0.003000, 0.160000))

        return segs


    def scaleSeg(self, segs, segid:str, buildp, stab, basep, tgtp, final):

        seg = segs[segid]
        if 'build' in segid :
            deltax = segs['buildstab']['sdx_stop'] - segs['buildp']['sdx_start']
            deltay = 1
            offsetx = 0
            offsety = basep
            y_gain = tgtp - basep

            x_seed = seg['sdx_stop'] - seg['sdx_start']
            x_time = (buildp * x_seed) / deltax

            x_steps = int(x_time * 4)
            x_gain = x_time / x_seed

        elif 'stabjump' in segid :
            seg['x'] = [segs['buildstab']['x'][-1],segs['stabinit']['x'][0]]
            seg['y'] = [segs['buildstab']['y'][-1],segs['stabinit']['y'][0
            ]]
            return True

        else:
            deltax = segs['stabend']['sdx_stop'] - segs['stabjump']['sdx_start']
            deltay = 1

            y_gain = tgtp - basep
            if 'stabend' in segid :
                polyx = seg['poly']
                data = polyx(1)
                y = self.scaleAxis(data, (basep, y_gain))
                offsety = (final -y) + basep
                segs['stabend']['yoffset'] = offsety
            else:
                offsety = segs['stabend']['yoffset']

            x_seed = seg['sdx_stop'] - seg['sdx_start']
            x_time = (stab * x_seed) / deltax
            x_steps = int(x_time * 4)
            x_gain = x_time / x_seed
            offsetx = (buildp+stab) - x_gain

        xp0 = np.linspace(seg['sdx_start'], seg['sdx_stop'], x_steps)
        polyx = seg['poly']
        data = polyx(xp0)

        x = self.scaleAxis(xp0, (offsetx, x_gain))
        y = self.scaleAxis(data, (offsety, y_gain))

        seg['x'] = x
        seg['y'] = y

        return True


    def scaleSegs(self, segs, buildp, stab, basep, tgtp, final):

        self.scaleSeg(segs, 'buildp', buildp, stab, basep, tgtp, final)
        self.scaleSeg(segs, 'buildstab', buildp, stab, basep, tgtp, final)

        self.scaleSeg(segs, 'stabend', buildp, stab, basep, tgtp, final)
        self.scaleSeg(segs, 'stabinit', buildp, stab, basep, tgtp, final)
        self.scaleSeg(segs, 'stabjump', buildp, stab, basep, tgtp, final)

        return True


    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))



    def showStatus(self):

        cv, brks = self.getSeed2()
        # segs = self.getSegments(cv, brks)
        segs = self.getCoeffs()

        fig, ax = plt.subplots(constrained_layout=True)
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        for idx, seg in enumerate(segs.values()):
            logger.info(self.showPoly(seg[0], "Seg poly {:2d} from {:9.6f} to {:9.6f} is : ".format(idx, seg[1], seg[2])))

        ax.set_ylabel('Pressão')
        # ax.plot(np.array(cv[:,0]), np.array(cv[:,1]), '.')

        self.scaleSegs(segs, 10, 33, 31.8242, 63.627, 48.997)
        # self.scaleSegs(segs, 11, 302, 48.997, 61.369, 54.7245)

        self.segs = segs

        for pseg in segs.values():
            ax.plot(pseg['x'], pseg['y'], '-')

        # plt.axvline (segs['buildstab']['sdx_stop'], label="Achieved")
        # plt.legend()

        # self.listPoints()

        # upper = ptsy[np.argmax(ptsy)]
        # lower = ptsy[np.argmin(ptsy)]
        # plt.ylim(lower-5, upper+5)

        # daxis = ax.twinx()
        # daxis.set_ylim(-0.8, 0.5)

        points = self.mergePoints()





        ptts = np.array([x for x in points if x[0] > 0.1])
        ptsx = np.array(ptts[:,0])
        ptsy = np.array(ptts[:,1])



        # func = TestFrame1.fitPoly3
        func = TestFrame1.fitAsynt
        popt = self.getFit(ptsx, ptsy, func)
        pfit = [func(x, *popt) for x in ptsx]
        ax.plot(ptsx, func(ptsx, *popt), '-')


        # dif = np.diff(pfit)
        # dy = np.array(dif[0])
        # dy = np.append(dy, dif)
        # # dy = np.gradient(ptsy, 4)
        # c = ptsx[ptsx > 0]
        #
        # # daxis.plot(c, dy, '-')


        plt.show()

        # logger.info(self.listStabPoly(segs, c, ptsy, dy))

        return

    def fitAsynt(x, a):
        return (10/x * -1) + 10

    def fitPoly2(x, a,b,c):
        return (a * (x*x)) + (b * x) + c

    def fitPoly3(x, a,b,c,d):
        return (a * (x*x*x)) + (b * (x*x)) + (c * x) + d

    def fitPoly4(x, a,b,c,d,e):
        return (a * (x*x*x*x)) + (b * (x*x*x)) + (c * (x*x)) + (d * x) + e

    def getFit (self, xdata, ydata, func):
        popt, pcov = curve_fit(func, xdata, ydata, method='lm')
        return popt



    def listStabPoly(self, segs, x, data, dif):

        sb = utils.StringBuilder('\n')

        sb+= ' Stabdata_start : {:9.6f} \n'.format( segs['stabjump']['x'][0])

        for idx, xdata in enumerate(x):
            sb+= ' {:3d} {:1.3f} {:9.6f} {:9.6f} \n'.format(idx, xdata,data[idx], dif[idx])
        return str(sb)


    def mergePoints(self):

        tstamp = 0
        pts = list()
        for pseg in self.segs.values():
            for idx, value in enumerate(pseg['y']):
                pts.append([(tstamp*250)/1000, value, pseg['x'][idx]])
                tstamp = tstamp+1
        return np.array(pts)


    def listPoints(self):

        sb = utils.StringBuilder('\n')
        tstamp = 0
        for pseg in self.segs.values():
            for value in pseg['y']:
                sb+= '{:8d} : {:9.6f}\n'.format(tstamp*250, value)
                tstamp = tstamp+1

        list1 = str(sb)
        logger.info('Curve computed {:4d} points: {}'.format(tstamp, list1))


    def getSeed2(self):
        cv = np.array([
            [  0.003 ,   0.003  ,1],
            [ .04 ,  .06  ,1],
            [ .14 ,  .80  ,1],
            [ .15 ,  .88  ,1],
            [ .16 ,  .96  ,10],
            [ .20 ,  .98  ,1],
            [ .24 ,  .99  ,1],
            [ .30 ,  1.   ,10],
            [ .32 ,  .60  ,2],
            [ .36 ,  .50  ,8],
            [ .40 ,  .45  ,10],
            [ .50 ,  .42  ,1],
            [ .60 ,  .40  ,1],
            [ .70 ,  .372 ,1],
            [ .80 ,  .352 ,1],
            [ .90 ,  .34  ,1],
            [ 1. ,   .328 ,1]
        ])
        return cv, ((0,4,2), (4,7,2), (7,8,1), (8,10,2), (10,16,3))



    def getSeed1(self):
        cv = np.array([
            [  0. ,   0.  ,1],
            [ .04 ,  .06  ,1],
            [ .14 ,  .80  ,1],
            [ .16 ,  .96  ,1],
            [ .20 ,  .98  ,1],
            [ .24 ,  .99  ,1],
            [ .30 ,  1.   ,1],
            [ .32 ,  .50  ,100],
            [ .40 ,  .45  ,1],
            [ .50 ,  .42  ,1],
            [ .60 ,  .40  ,1],
            [ .70 ,  .372 ,1],
            [ .80 ,  .352 ,1],
            [ .90 ,  .34  ,1],
            [ 1. ,   .328 ,1]
        ])
        return cv, ((0,3,2), (3,7,2), (7,8,1), (7,14,1))




    def bokeh1(self):

        x = [x*0.005 for x in range(0, 201)]

        source = ColumnDataSource(data=dict(x=x, y=x))

        plot = figure(plot_width=400, plot_height=400)
        plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

        slider = Slider(start=0.1, end=6, value=1, step=.1, title="power")

        update_curve = CustomJS(args=dict(source=source, slider=slider), code="""
            var data = source.data;
            var f = slider.value;
            x = data['x']
            y = data['y']
            for (i = 0; i < x.length; i++) {
                y[i] = Math.pow(x[i], f)
            }
            
            // necessary becasue we mutated source.data in-place
            source.change.emit();
        """)
        slider.js_on_change('value', update_curve)


        show(column(slider, plot))



    def bokeh2(self):

        # create some widgets
        button = Button(label="Say HI")
        input = TextInput(value="Bokeh")
        output = Paragraph()

        # add a callback to a widget
        def update():
            output.text = "Hello, " + input.value
        button.on_click(update)

        # create a layout for everything
        layout = column(button, input, output)

        # add the layout to curdoc
        curdoc().add_root(layout)
