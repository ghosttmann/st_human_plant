#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on November 17, 2025, at 18:33
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'builder_human_plant'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\st_human_plant\\builder_human_plant_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('video_cıkıs') is None:
        # initialise video_cıkıs
        video_cıkıs = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='video_cıkıs',
        )
    if deviceManager.getDevice('cikis') is None:
        # initialise cikis
        cikis = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cikis',
        )
    if deviceManager.getDevice('isinma_cikis') is None:
        # initialise isinma_cikis
        isinma_cikis = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='isinma_cikis',
        )
    if deviceManager.getDevice('res_asama_2') is None:
        # initialise res_asama_2
        res_asama_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='res_asama_2',
        )
    if deviceManager.getDevice('asama_2_res') is None:
        # initialise asama_2_res
        asama_2_res = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='asama_2_res',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "age" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Yasınız nedir? \n1=18  2=19  3=20  4=21\n5=22  6=23 7=24 8= 25  9=diğer',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "uni" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Üniversiteniz ÇOMÜ ise "1", değilse "2" seçiniz.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "video_asama" ---
    video_bilgi = visual.TextStim(win=win, name='video_bilgi',
        text='Video izleyeceksiniz',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    video = visual.MovieStim(
        win, name='video',
        filename='human_plant.mp4', movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(1, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-1
    )
    movie_devam_bilgi = visual.TextStim(win=win, name='movie_devam_bilgi',
        text='devam etmek için\n"boşluk" tuşuna basınız',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    video_cıkıs = keyboard.Keyboard(deviceName='video_cıkıs')
    
    # --- Initialize components for Routine "baslangıc" ---
    isinma_bilgi = visual.TextStim(win=win, name='isinma_bilgi',
        text='Isınma aşamasına hoş geldiniz.\nDevam etmek için boşluk tuşuna basınız',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cikis = keyboard.Keyboard(deviceName='cikis')
    
    # --- Initialize components for Routine "isinma_st" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_isinma = visual.ImageStim(
        win=win,
        name='image_isinma', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    disp_word = visual.TextStim(win=win, name='disp_word',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='red', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    isinma_cikis = keyboard.Keyboard(deviceName='isinma_cikis')
    
    # --- Initialize components for Routine "asama_1" ---
    asama_1_bilgi = visual.TextStim(win=win, name='asama_1_bilgi',
        text='Asama 1 birazdan başlayacaktır. \nGörev aynı.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "asama_1_st" ---
    fix_asama_1 = visual.TextStim(win=win, name='fix_asama_1',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_asama_2 = visual.ImageStim(
        win=win,
        name='image_asama_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    disp_word_asama_2 = visual.TextStim(win=win, name='disp_word_asama_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='red', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    res_asama_2 = keyboard.Keyboard(deviceName='res_asama_2')
    
    # --- Initialize components for Routine "asama_2" ---
    asama_2_bilgi = visual.TextStim(win=win, name='asama_2_bilgi',
        text='asama 2 başlıyor',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "asama_2_st" ---
    fix_asama_2 = visual.TextStim(win=win, name='fix_asama_2',
        text='+\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    asama_2_res = keyboard.Keyboard(deviceName='asama_2_res')
    
    # --- Initialize components for Routine "btis" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Katılım gösterdiğiniz için teşekkür ederim. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "age" ---
    # create an object to store info about Routine age
    age = data.Routine(
        name='age',
        components=[text_3, key_resp_2],
    )
    age.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for age
    age.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    age.tStart = globalClock.getTime(format='float')
    age.status = STARTED
    thisExp.addData('age.started', age.tStart)
    age.maxDuration = None
    # keep track of which components have finished
    ageComponents = age.components
    for thisComponent in age.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "age" ---
    age.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=["1","2","3","4","5","6","7","8","9"], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=age,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            age.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in age.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "age" ---
    for thisComponent in age.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for age
    age.tStop = globalClock.getTime(format='float')
    age.tStopRefresh = tThisFlipGlobal
    thisExp.addData('age.stopped', age.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "age" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "uni" ---
    # create an object to store info about Routine uni
    uni = data.Routine(
        name='uni',
        components=[text_4, key_resp_3],
    )
    uni.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for uni
    uni.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    uni.tStart = globalClock.getTime(format='float')
    uni.status = STARTED
    thisExp.addData('uni.started', uni.tStart)
    uni.maxDuration = None
    # keep track of which components have finished
    uniComponents = uni.components
    for thisComponent in uni.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "uni" ---
    uni.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=["1","0"], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=uni,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            uni.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in uni.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "uni" ---
    for thisComponent in uni.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for uni
    uni.tStop = globalClock.getTime(format='float')
    uni.tStopRefresh = tThisFlipGlobal
    thisExp.addData('uni.stopped', uni.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "uni" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "video_asama" ---
    # create an object to store info about Routine video_asama
    video_asama = data.Routine(
        name='video_asama',
        components=[video_bilgi, video, movie_devam_bilgi, video_cıkıs],
    )
    video_asama.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for video_cıkıs
    video_cıkıs.keys = []
    video_cıkıs.rt = []
    _video_cıkıs_allKeys = []
    # store start times for video_asama
    video_asama.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    video_asama.tStart = globalClock.getTime(format='float')
    video_asama.status = STARTED
    thisExp.addData('video_asama.started', video_asama.tStart)
    video_asama.maxDuration = None
    # keep track of which components have finished
    video_asamaComponents = video_asama.components
    for thisComponent in video_asama.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "video_asama" ---
    video_asama.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *video_bilgi* updates
        
        # if video_bilgi is starting this frame...
        if video_bilgi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            video_bilgi.frameNStart = frameN  # exact frame index
            video_bilgi.tStart = t  # local t and not account for scr refresh
            video_bilgi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video_bilgi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video_bilgi.started')
            # update status
            video_bilgi.status = STARTED
            video_bilgi.setAutoDraw(True)
        
        # if video_bilgi is active this frame...
        if video_bilgi.status == STARTED:
            # update params
            pass
        
        # if video_bilgi is stopping this frame...
        if video_bilgi.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video_bilgi.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                video_bilgi.tStop = t  # not accounting for scr refresh
                video_bilgi.tStopRefresh = tThisFlipGlobal  # on global time
                video_bilgi.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video_bilgi.stopped')
                # update status
                video_bilgi.status = FINISHED
                video_bilgi.setAutoDraw(False)
        
        # *video* updates
        
        # if video is starting this frame...
        if video.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            video.frameNStart = frameN  # exact frame index
            video.tStart = t  # local t and not account for scr refresh
            video.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video.started')
            # update status
            video.status = STARTED
            video.setAutoDraw(True)
            video.play()
        
        # if video is stopping this frame...
        if video.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video.tStartRefresh + 320-frameTolerance or video.isFinished:
                # keep track of stop time/frame for later
                video.tStop = t  # not accounting for scr refresh
                video.tStopRefresh = tThisFlipGlobal  # on global time
                video.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video.stopped')
                # update status
                video.status = FINISHED
                video.setAutoDraw(False)
                video.stop()
        
        # *movie_devam_bilgi* updates
        
        # if movie_devam_bilgi is starting this frame...
        if movie_devam_bilgi.status == NOT_STARTED and tThisFlip >= 325-frameTolerance:
            # keep track of start time/frame for later
            movie_devam_bilgi.frameNStart = frameN  # exact frame index
            movie_devam_bilgi.tStart = t  # local t and not account for scr refresh
            movie_devam_bilgi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(movie_devam_bilgi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'movie_devam_bilgi.started')
            # update status
            movie_devam_bilgi.status = STARTED
            movie_devam_bilgi.setAutoDraw(True)
        
        # if movie_devam_bilgi is active this frame...
        if movie_devam_bilgi.status == STARTED:
            # update params
            pass
        
        # *video_cıkıs* updates
        waitOnFlip = False
        
        # if video_cıkıs is starting this frame...
        if video_cıkıs.status == NOT_STARTED and tThisFlip >= 325-frameTolerance:
            # keep track of start time/frame for later
            video_cıkıs.frameNStart = frameN  # exact frame index
            video_cıkıs.tStart = t  # local t and not account for scr refresh
            video_cıkıs.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video_cıkıs, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video_cıkıs.started')
            # update status
            video_cıkıs.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(video_cıkıs.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(video_cıkıs.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if video_cıkıs.status == STARTED and not waitOnFlip:
            theseKeys = video_cıkıs.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _video_cıkıs_allKeys.extend(theseKeys)
            if len(_video_cıkıs_allKeys):
                video_cıkıs.keys = _video_cıkıs_allKeys[-1].name  # just the last key pressed
                video_cıkıs.rt = _video_cıkıs_allKeys[-1].rt
                video_cıkıs.duration = _video_cıkıs_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=video_asama,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            video_asama.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in video_asama.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "video_asama" ---
    for thisComponent in video_asama.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for video_asama
    video_asama.tStop = globalClock.getTime(format='float')
    video_asama.tStopRefresh = tThisFlipGlobal
    thisExp.addData('video_asama.stopped', video_asama.tStop)
    video.stop()  # ensure movie has stopped at end of Routine
    # check responses
    if video_cıkıs.keys in ['', [], None]:  # No response was made
        video_cıkıs.keys = None
    thisExp.addData('video_cıkıs.keys',video_cıkıs.keys)
    if video_cıkıs.keys != None:  # we had a response
        thisExp.addData('video_cıkıs.rt', video_cıkıs.rt)
        thisExp.addData('video_cıkıs.duration', video_cıkıs.duration)
    thisExp.nextEntry()
    # the Routine "video_asama" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "baslangıc" ---
    # create an object to store info about Routine baslangıc
    baslangıc = data.Routine(
        name='baslangıc',
        components=[isinma_bilgi, cikis],
    )
    baslangıc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for cikis
    cikis.keys = []
    cikis.rt = []
    _cikis_allKeys = []
    # store start times for baslangıc
    baslangıc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    baslangıc.tStart = globalClock.getTime(format='float')
    baslangıc.status = STARTED
    thisExp.addData('baslangıc.started', baslangıc.tStart)
    baslangıc.maxDuration = None
    # keep track of which components have finished
    baslangıcComponents = baslangıc.components
    for thisComponent in baslangıc.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "baslangıc" ---
    baslangıc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *isinma_bilgi* updates
        
        # if isinma_bilgi is starting this frame...
        if isinma_bilgi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            isinma_bilgi.frameNStart = frameN  # exact frame index
            isinma_bilgi.tStart = t  # local t and not account for scr refresh
            isinma_bilgi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(isinma_bilgi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'isinma_bilgi.started')
            # update status
            isinma_bilgi.status = STARTED
            isinma_bilgi.setAutoDraw(True)
        
        # if isinma_bilgi is active this frame...
        if isinma_bilgi.status == STARTED:
            # update params
            pass
        
        # *cikis* updates
        waitOnFlip = False
        
        # if cikis is starting this frame...
        if cikis.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cikis.frameNStart = frameN  # exact frame index
            cikis.tStart = t  # local t and not account for scr refresh
            cikis.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cikis, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cikis.started')
            # update status
            cikis.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(cikis.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(cikis.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if cikis.status == STARTED and not waitOnFlip:
            theseKeys = cikis.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _cikis_allKeys.extend(theseKeys)
            if len(_cikis_allKeys):
                cikis.keys = _cikis_allKeys[-1].name  # just the last key pressed
                cikis.rt = _cikis_allKeys[-1].rt
                cikis.duration = _cikis_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=baslangıc,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            baslangıc.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in baslangıc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "baslangıc" ---
    for thisComponent in baslangıc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for baslangıc
    baslangıc.tStop = globalClock.getTime(format='float')
    baslangıc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('baslangıc.stopped', baslangıc.tStop)
    # check responses
    if cikis.keys in ['', [], None]:  # No response was made
        cikis.keys = None
    thisExp.addData('cikis.keys',cikis.keys)
    if cikis.keys != None:  # we had a response
        thisExp.addData('cikis.rt', cikis.rt)
        thisExp.addData('cikis.duration', cikis.duration)
    thisExp.nextEntry()
    # the Routine "baslangıc" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('ısınma_kosullar.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "isinma_st" ---
        # create an object to store info about Routine isinma_st
        isinma_st = data.Routine(
            name='isinma_st',
            components=[fixation, image_isinma, disp_word, isinma_cikis],
        )
        isinma_st.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_isinma.setImage(imageFile)
        disp_word.setText(display_word)
        # create starting attributes for isinma_cikis
        isinma_cikis.keys = []
        isinma_cikis.rt = []
        _isinma_cikis_allKeys = []
        # store start times for isinma_st
        isinma_st.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        isinma_st.tStart = globalClock.getTime(format='float')
        isinma_st.status = STARTED
        thisExp.addData('isinma_st.started', isinma_st.tStart)
        isinma_st.maxDuration = None
        # keep track of which components have finished
        isinma_stComponents = isinma_st.components
        for thisComponent in isinma_st.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "isinma_st" ---
        isinma_st.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *image_isinma* updates
            
            # if image_isinma is starting this frame...
            if image_isinma.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                image_isinma.frameNStart = frameN  # exact frame index
                image_isinma.tStart = t  # local t and not account for scr refresh
                image_isinma.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_isinma, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_isinma.started')
                # update status
                image_isinma.status = STARTED
                image_isinma.setAutoDraw(True)
            
            # if image_isinma is active this frame...
            if image_isinma.status == STARTED:
                # update params
                pass
            
            # if image_isinma is stopping this frame...
            if image_isinma.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_isinma.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_isinma.tStop = t  # not accounting for scr refresh
                    image_isinma.tStopRefresh = tThisFlipGlobal  # on global time
                    image_isinma.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_isinma.stopped')
                    # update status
                    image_isinma.status = FINISHED
                    image_isinma.setAutoDraw(False)
            
            # *disp_word* updates
            
            # if disp_word is starting this frame...
            if disp_word.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                disp_word.frameNStart = frameN  # exact frame index
                disp_word.tStart = t  # local t and not account for scr refresh
                disp_word.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(disp_word, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'disp_word.started')
                # update status
                disp_word.status = STARTED
                disp_word.setAutoDraw(True)
            
            # if disp_word is active this frame...
            if disp_word.status == STARTED:
                # update params
                pass
            
            # if disp_word is stopping this frame...
            if disp_word.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > disp_word.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    disp_word.tStop = t  # not accounting for scr refresh
                    disp_word.tStopRefresh = tThisFlipGlobal  # on global time
                    disp_word.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'disp_word.stopped')
                    # update status
                    disp_word.status = FINISHED
                    disp_word.setAutoDraw(False)
            
            # *isinma_cikis* updates
            waitOnFlip = False
            
            # if isinma_cikis is starting this frame...
            if isinma_cikis.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                isinma_cikis.frameNStart = frameN  # exact frame index
                isinma_cikis.tStart = t  # local t and not account for scr refresh
                isinma_cikis.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(isinma_cikis, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'isinma_cikis.started')
                # update status
                isinma_cikis.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(isinma_cikis.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(isinma_cikis.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if isinma_cikis is stopping this frame...
            if isinma_cikis.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > isinma_cikis.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    isinma_cikis.tStop = t  # not accounting for scr refresh
                    isinma_cikis.tStopRefresh = tThisFlipGlobal  # on global time
                    isinma_cikis.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'isinma_cikis.stopped')
                    # update status
                    isinma_cikis.status = FINISHED
                    isinma_cikis.status = FINISHED
            if isinma_cikis.status == STARTED and not waitOnFlip:
                theseKeys = isinma_cikis.getKeys(keyList=["f","j"], ignoreKeys=["escape"], waitRelease=False)
                _isinma_cikis_allKeys.extend(theseKeys)
                if len(_isinma_cikis_allKeys):
                    isinma_cikis.keys = _isinma_cikis_allKeys[-1].name  # just the last key pressed
                    isinma_cikis.rt = _isinma_cikis_allKeys[-1].rt
                    isinma_cikis.duration = _isinma_cikis_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=isinma_st,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                isinma_st.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in isinma_st.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "isinma_st" ---
        for thisComponent in isinma_st.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for isinma_st
        isinma_st.tStop = globalClock.getTime(format='float')
        isinma_st.tStopRefresh = tThisFlipGlobal
        thisExp.addData('isinma_st.stopped', isinma_st.tStop)
        # check responses
        if isinma_cikis.keys in ['', [], None]:  # No response was made
            isinma_cikis.keys = None
        trials.addData('isinma_cikis.keys',isinma_cikis.keys)
        if isinma_cikis.keys != None:  # we had a response
            trials.addData('isinma_cikis.rt', isinma_cikis.rt)
            trials.addData('isinma_cikis.duration', isinma_cikis.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if isinma_st.maxDurationReached:
            routineTimer.addTime(-isinma_st.maxDuration)
        elif isinma_st.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "asama_1" ---
    # create an object to store info about Routine asama_1
    asama_1 = data.Routine(
        name='asama_1',
        components=[asama_1_bilgi],
    )
    asama_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for asama_1
    asama_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    asama_1.tStart = globalClock.getTime(format='float')
    asama_1.status = STARTED
    thisExp.addData('asama_1.started', asama_1.tStart)
    asama_1.maxDuration = None
    # keep track of which components have finished
    asama_1Components = asama_1.components
    for thisComponent in asama_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "asama_1" ---
    asama_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *asama_1_bilgi* updates
        
        # if asama_1_bilgi is starting this frame...
        if asama_1_bilgi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            asama_1_bilgi.frameNStart = frameN  # exact frame index
            asama_1_bilgi.tStart = t  # local t and not account for scr refresh
            asama_1_bilgi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(asama_1_bilgi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'asama_1_bilgi.started')
            # update status
            asama_1_bilgi.status = STARTED
            asama_1_bilgi.setAutoDraw(True)
        
        # if asama_1_bilgi is active this frame...
        if asama_1_bilgi.status == STARTED:
            # update params
            pass
        
        # if asama_1_bilgi is stopping this frame...
        if asama_1_bilgi.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > asama_1_bilgi.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                asama_1_bilgi.tStop = t  # not accounting for scr refresh
                asama_1_bilgi.tStopRefresh = tThisFlipGlobal  # on global time
                asama_1_bilgi.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'asama_1_bilgi.stopped')
                # update status
                asama_1_bilgi.status = FINISHED
                asama_1_bilgi.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=asama_1,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            asama_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in asama_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "asama_1" ---
    for thisComponent in asama_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for asama_1
    asama_1.tStop = globalClock.getTime(format='float')
    asama_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('asama_1.stopped', asama_1.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if asama_1.maxDurationReached:
        routineTimer.addTime(-asama_1.maxDuration)
    elif asama_1.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('kosullar.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        trials_2.status = STARTED
        if hasattr(thisTrial_2, 'status'):
            thisTrial_2.status = STARTED
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "asama_1_st" ---
        # create an object to store info about Routine asama_1_st
        asama_1_st = data.Routine(
            name='asama_1_st',
            components=[fix_asama_1, image_asama_2, disp_word_asama_2, res_asama_2],
        )
        asama_1_st.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_asama_2.setImage(imageFile)
        disp_word_asama_2.setText(display_word)
        # create starting attributes for res_asama_2
        res_asama_2.keys = []
        res_asama_2.rt = []
        _res_asama_2_allKeys = []
        # store start times for asama_1_st
        asama_1_st.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        asama_1_st.tStart = globalClock.getTime(format='float')
        asama_1_st.status = STARTED
        thisExp.addData('asama_1_st.started', asama_1_st.tStart)
        asama_1_st.maxDuration = None
        # keep track of which components have finished
        asama_1_stComponents = asama_1_st.components
        for thisComponent in asama_1_st.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "asama_1_st" ---
        asama_1_st.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_2, 'status') and thisTrial_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_asama_1* updates
            
            # if fix_asama_1 is starting this frame...
            if fix_asama_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_asama_1.frameNStart = frameN  # exact frame index
                fix_asama_1.tStart = t  # local t and not account for scr refresh
                fix_asama_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_asama_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_asama_1.started')
                # update status
                fix_asama_1.status = STARTED
                fix_asama_1.setAutoDraw(True)
            
            # if fix_asama_1 is active this frame...
            if fix_asama_1.status == STARTED:
                # update params
                pass
            
            # if fix_asama_1 is stopping this frame...
            if fix_asama_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_asama_1.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_asama_1.tStop = t  # not accounting for scr refresh
                    fix_asama_1.tStopRefresh = tThisFlipGlobal  # on global time
                    fix_asama_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_asama_1.stopped')
                    # update status
                    fix_asama_1.status = FINISHED
                    fix_asama_1.setAutoDraw(False)
            
            # *image_asama_2* updates
            
            # if image_asama_2 is starting this frame...
            if image_asama_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                image_asama_2.frameNStart = frameN  # exact frame index
                image_asama_2.tStart = t  # local t and not account for scr refresh
                image_asama_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_asama_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_asama_2.started')
                # update status
                image_asama_2.status = STARTED
                image_asama_2.setAutoDraw(True)
            
            # if image_asama_2 is active this frame...
            if image_asama_2.status == STARTED:
                # update params
                pass
            
            # if image_asama_2 is stopping this frame...
            if image_asama_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_asama_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_asama_2.tStop = t  # not accounting for scr refresh
                    image_asama_2.tStopRefresh = tThisFlipGlobal  # on global time
                    image_asama_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_asama_2.stopped')
                    # update status
                    image_asama_2.status = FINISHED
                    image_asama_2.setAutoDraw(False)
            
            # *disp_word_asama_2* updates
            
            # if disp_word_asama_2 is starting this frame...
            if disp_word_asama_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                disp_word_asama_2.frameNStart = frameN  # exact frame index
                disp_word_asama_2.tStart = t  # local t and not account for scr refresh
                disp_word_asama_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(disp_word_asama_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'disp_word_asama_2.started')
                # update status
                disp_word_asama_2.status = STARTED
                disp_word_asama_2.setAutoDraw(True)
            
            # if disp_word_asama_2 is active this frame...
            if disp_word_asama_2.status == STARTED:
                # update params
                pass
            
            # if disp_word_asama_2 is stopping this frame...
            if disp_word_asama_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > disp_word_asama_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    disp_word_asama_2.tStop = t  # not accounting for scr refresh
                    disp_word_asama_2.tStopRefresh = tThisFlipGlobal  # on global time
                    disp_word_asama_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'disp_word_asama_2.stopped')
                    # update status
                    disp_word_asama_2.status = FINISHED
                    disp_word_asama_2.setAutoDraw(False)
            
            # *res_asama_2* updates
            waitOnFlip = False
            
            # if res_asama_2 is starting this frame...
            if res_asama_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                res_asama_2.frameNStart = frameN  # exact frame index
                res_asama_2.tStart = t  # local t and not account for scr refresh
                res_asama_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(res_asama_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'res_asama_2.started')
                # update status
                res_asama_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(res_asama_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(res_asama_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if res_asama_2 is stopping this frame...
            if res_asama_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > res_asama_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    res_asama_2.tStop = t  # not accounting for scr refresh
                    res_asama_2.tStopRefresh = tThisFlipGlobal  # on global time
                    res_asama_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'res_asama_2.stopped')
                    # update status
                    res_asama_2.status = FINISHED
                    res_asama_2.status = FINISHED
            if res_asama_2.status == STARTED and not waitOnFlip:
                theseKeys = res_asama_2.getKeys(keyList=["f","j"], ignoreKeys=["escape"], waitRelease=False)
                _res_asama_2_allKeys.extend(theseKeys)
                if len(_res_asama_2_allKeys):
                    res_asama_2.keys = _res_asama_2_allKeys[-1].name  # just the last key pressed
                    res_asama_2.rt = _res_asama_2_allKeys[-1].rt
                    res_asama_2.duration = _res_asama_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=asama_1_st,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                asama_1_st.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in asama_1_st.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "asama_1_st" ---
        for thisComponent in asama_1_st.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for asama_1_st
        asama_1_st.tStop = globalClock.getTime(format='float')
        asama_1_st.tStopRefresh = tThisFlipGlobal
        thisExp.addData('asama_1_st.stopped', asama_1_st.tStop)
        # check responses
        if res_asama_2.keys in ['', [], None]:  # No response was made
            res_asama_2.keys = None
        trials_2.addData('res_asama_2.keys',res_asama_2.keys)
        if res_asama_2.keys != None:  # we had a response
            trials_2.addData('res_asama_2.rt', res_asama_2.rt)
            trials_2.addData('res_asama_2.duration', res_asama_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if asama_1_st.maxDurationReached:
            routineTimer.addTime(-asama_1_st.maxDuration)
        elif asama_1_st.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        # mark thisTrial_2 as finished
        if hasattr(thisTrial_2, 'status'):
            thisTrial_2.status = FINISHED
        # if awaiting a pause, pause now
        if trials_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_2'
    trials_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "asama_2" ---
    # create an object to store info about Routine asama_2
    asama_2 = data.Routine(
        name='asama_2',
        components=[asama_2_bilgi],
    )
    asama_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for asama_2
    asama_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    asama_2.tStart = globalClock.getTime(format='float')
    asama_2.status = STARTED
    thisExp.addData('asama_2.started', asama_2.tStart)
    asama_2.maxDuration = None
    # keep track of which components have finished
    asama_2Components = asama_2.components
    for thisComponent in asama_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "asama_2" ---
    asama_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *asama_2_bilgi* updates
        
        # if asama_2_bilgi is starting this frame...
        if asama_2_bilgi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            asama_2_bilgi.frameNStart = frameN  # exact frame index
            asama_2_bilgi.tStart = t  # local t and not account for scr refresh
            asama_2_bilgi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(asama_2_bilgi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'asama_2_bilgi.started')
            # update status
            asama_2_bilgi.status = STARTED
            asama_2_bilgi.setAutoDraw(True)
        
        # if asama_2_bilgi is active this frame...
        if asama_2_bilgi.status == STARTED:
            # update params
            pass
        
        # if asama_2_bilgi is stopping this frame...
        if asama_2_bilgi.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > asama_2_bilgi.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                asama_2_bilgi.tStop = t  # not accounting for scr refresh
                asama_2_bilgi.tStopRefresh = tThisFlipGlobal  # on global time
                asama_2_bilgi.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'asama_2_bilgi.stopped')
                # update status
                asama_2_bilgi.status = FINISHED
                asama_2_bilgi.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=asama_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            asama_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in asama_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "asama_2" ---
    for thisComponent in asama_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for asama_2
    asama_2.tStop = globalClock.getTime(format='float')
    asama_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('asama_2.stopped', asama_2.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if asama_2.maxDurationReached:
        routineTimer.addTime(-asama_2.maxDuration)
    elif asama_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler2(
        name='trials_3',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('kosullar_2.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_3 in trials_3:
        trials_3.status = STARTED
        if hasattr(thisTrial_3, 'status'):
            thisTrial_3.status = STARTED
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "asama_2_st" ---
        # create an object to store info about Routine asama_2_st
        asama_2_st = data.Routine(
            name='asama_2_st',
            components=[fix_asama_2, image, text, asama_2_res],
        )
        asama_2_st.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image.setImage(imageFile)
        text.setText(display_word
        )
        # create starting attributes for asama_2_res
        asama_2_res.keys = []
        asama_2_res.rt = []
        _asama_2_res_allKeys = []
        # store start times for asama_2_st
        asama_2_st.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        asama_2_st.tStart = globalClock.getTime(format='float')
        asama_2_st.status = STARTED
        thisExp.addData('asama_2_st.started', asama_2_st.tStart)
        asama_2_st.maxDuration = None
        # keep track of which components have finished
        asama_2_stComponents = asama_2_st.components
        for thisComponent in asama_2_st.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "asama_2_st" ---
        asama_2_st.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial_3, 'status') and thisTrial_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_asama_2* updates
            
            # if fix_asama_2 is starting this frame...
            if fix_asama_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_asama_2.frameNStart = frameN  # exact frame index
                fix_asama_2.tStart = t  # local t and not account for scr refresh
                fix_asama_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_asama_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_asama_2.started')
                # update status
                fix_asama_2.status = STARTED
                fix_asama_2.setAutoDraw(True)
            
            # if fix_asama_2 is active this frame...
            if fix_asama_2.status == STARTED:
                # update params
                pass
            
            # if fix_asama_2 is stopping this frame...
            if fix_asama_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_asama_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_asama_2.tStop = t  # not accounting for scr refresh
                    fix_asama_2.tStopRefresh = tThisFlipGlobal  # on global time
                    fix_asama_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_asama_2.stopped')
                    # update status
                    fix_asama_2.status = FINISHED
                    fix_asama_2.setAutoDraw(False)
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # *asama_2_res* updates
            waitOnFlip = False
            
            # if asama_2_res is starting this frame...
            if asama_2_res.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                asama_2_res.frameNStart = frameN  # exact frame index
                asama_2_res.tStart = t  # local t and not account for scr refresh
                asama_2_res.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(asama_2_res, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'asama_2_res.started')
                # update status
                asama_2_res.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(asama_2_res.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(asama_2_res.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if asama_2_res is stopping this frame...
            if asama_2_res.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > asama_2_res.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    asama_2_res.tStop = t  # not accounting for scr refresh
                    asama_2_res.tStopRefresh = tThisFlipGlobal  # on global time
                    asama_2_res.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'asama_2_res.stopped')
                    # update status
                    asama_2_res.status = FINISHED
                    asama_2_res.status = FINISHED
            if asama_2_res.status == STARTED and not waitOnFlip:
                theseKeys = asama_2_res.getKeys(keyList=["f","j"], ignoreKeys=["escape"], waitRelease=False)
                _asama_2_res_allKeys.extend(theseKeys)
                if len(_asama_2_res_allKeys):
                    asama_2_res.keys = _asama_2_res_allKeys[-1].name  # just the last key pressed
                    asama_2_res.rt = _asama_2_res_allKeys[-1].rt
                    asama_2_res.duration = _asama_2_res_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=asama_2_st,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                asama_2_st.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in asama_2_st.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "asama_2_st" ---
        for thisComponent in asama_2_st.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for asama_2_st
        asama_2_st.tStop = globalClock.getTime(format='float')
        asama_2_st.tStopRefresh = tThisFlipGlobal
        thisExp.addData('asama_2_st.stopped', asama_2_st.tStop)
        # check responses
        if asama_2_res.keys in ['', [], None]:  # No response was made
            asama_2_res.keys = None
        trials_3.addData('asama_2_res.keys',asama_2_res.keys)
        if asama_2_res.keys != None:  # we had a response
            trials_3.addData('asama_2_res.rt', asama_2_res.rt)
            trials_3.addData('asama_2_res.duration', asama_2_res.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if asama_2_st.maxDurationReached:
            routineTimer.addTime(-asama_2_st.maxDuration)
        elif asama_2_st.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        # mark thisTrial_3 as finished
        if hasattr(thisTrial_3, 'status'):
            thisTrial_3.status = FINISHED
        # if awaiting a pause, pause now
        if trials_3.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_3.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_3'
    trials_3.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "btis" ---
    # create an object to store info about Routine btis
    btis = data.Routine(
        name='btis',
        components=[text_2],
    )
    btis.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for btis
    btis.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    btis.tStart = globalClock.getTime(format='float')
    btis.status = STARTED
    thisExp.addData('btis.started', btis.tStart)
    btis.maxDuration = None
    # keep track of which components have finished
    btisComponents = btis.components
    for thisComponent in btis.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "btis" ---
    btis.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=btis,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            btis.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in btis.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "btis" ---
    for thisComponent in btis.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for btis
    btis.tStop = globalClock.getTime(format='float')
    btis.tStopRefresh = tThisFlipGlobal
    thisExp.addData('btis.stopped', btis.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if btis.maxDurationReached:
        routineTimer.addTime(-btis.maxDuration)
    elif btis.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
