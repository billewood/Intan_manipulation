# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:52:47 2017

@author: billewood
"""
import os
try:
    import wx
except ImportError:
    raise ImportError, "The wxPython module is required to run this program."
try:       
    from zeebeez.tdt2neo import stim
except ImportError:
    raise ImportError, "zeebeez.tdt2neo.stim module is required, it can be found at https://github.com/theunissenlab/zeebeez"   
    
class RHDImporter(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        #Unfortunately I can't get the filedialog open file and dir modules to work with default dirs
        self.defaultrhddir = '/auto/tdrive/billewood/intan data/LblY6074/RHD/'
        self.defaultcsvdir = '/auto/tdrive/billewood/intan data/LblY6074/pyoperant/'
        self.defaultstimdir = '/auto/tdrive/billewood/intan data/LblY6074/Stimuli/'
        self.defaultsavedir = '/auto/tdrive/billewood/intan data/'
        
        # GridBagSizer seems the easiest to work with
        grid = wx.GridBagSizer()#hgap=5, vgap=5)
        self.rhdfile = ''
        self.stimdir = ''
        self.stimfile = ''
        self.logfile = ''
        self.savedir = ''
        # Set up panel
        self.rhdlabel = wx.StaticText(self, label="RHD file not yet set ")
        grid.Add(self.rhdlabel, (0,3), (1,15), wx.ALIGN_LEFT)
        
        self.savedirlabel = wx.StaticText(self, label= "default save dir %s " % self.defaultsavedir)
        grid.Add(self.savedirlabel, (1,3), (1,15), wx.ALIGN_LEFT)
        
        self.stimdirlabel = wx.StaticText(self, label="default stim dir %s " % self.defaultstimdir)
        grid.Add(self.stimdirlabel, (3,3), (1,15), wx.ALIGN_LEFT)
        
        self.stimfilelabel = wx.StaticText(self, label="pyoperant stim file not set (if exists): ")
        grid.Add(self.stimfilelabel, (4,3), (1,15), wx.ALIGN_LEFT)
        
        self.stimsavefilenamelabel = wx.StaticText(self, label="Neosound DB filename not yet set ")
        grid.Add(self.stimsavefilenamelabel, (7,6), (1,15), wx.ALIGN_LEFT)

        self.stimsavefilenametextctrl = wx.TextCtrl(self, style = wx.TE_PROCESS_ENTER, size = (200,-1), value = "default filename")
        self.stimsavefilenametextctrl.Bind(wx.EVT_TEXT_ENTER, self.StimSaveFileNameButton, self.stimsavefilenametextctrl)        
        grid.Add(self.stimsavefilenametextctrl, (6,3), (1,15), wx.ALIGN_LEFT)
        
        # A multiline TextCtrl for displaying stim tags, not working
        self.stimtagdisplay = wx.TextCtrl(self, size=(300,300), style=wx.TE_MULTILINE | wx.TE_READONLY)
        grid.Add(self.stimtagdisplay, (10,3), (1,1), wx.ALIGN_LEFT)

        #Set up the buttons        
        # Open an RHD file button
        self.rhdbutton = wx.Button(self, -1, label = "Set RHD File")
        self.Bind(wx.EVT_BUTTON, self.ClickRHDFile, self.rhdbutton)
        grid.Add(self.rhdbutton, (0,0), (1,1), wx.EXPAND)
        
        # set stim dir
        self.savedirbutton = wx.Button(self, -1, label = "Set save dir")
        self.Bind(wx.EVT_BUTTON, self.ClickSaveDir, self.savedirbutton)
        grid.Add(self.savedirbutton, (1,0), (1,1), wx.EXPAND)
        
        # set stim audio file dir button
        self.stimdirbutton = wx.Button(self, -1, label = "Set audio stim dir (pyoperant)")
        self.Bind(wx.EVT_BUTTON, self.ClickStimDir, self.stimdirbutton)
        grid.Add(self.stimdirbutton, (3,0), (1,1), wx.EXPAND)
                
        # set stim file (pyoperant, .csv) button
        self.stimfilebutton = wx.Button(self, -1, label = "Set stim tags file (pyoperant, .csv) ")
        self.Bind(wx.EVT_BUTTON, self.ClickStimFile, self.stimfilebutton)
        grid.Add(self.stimfilebutton, (4,0), (1,1), wx.EXPAND)       
        
        self.stimsavefilenamebutton = wx.Button(self, -1, label = "Update neosound filename")
        self.Bind(wx.EVT_BUTTON, self.StimSaveFileNameButton, self.stimsavefilenamebutton)
        grid.Add(self.stimsavefilenamebutton, (7,3), (1,1))
        
        self.genstimsavefilenamebutton = wx.Button(self, -1, label = "Auto generate neosound filename")
        self.Bind(wx.EVT_BUTTON, self.GenStimSaveFileNameButton, self.genstimsavefilenamebutton)
        grid.Add(self.genstimsavefilenamebutton, (8,3), (1,1))   

        # Check stimuli tags, which imports just the playback stim tags
        # translates them to verify thigns are going well.
        self.makeneosoundbutton =wx.Button(self, label="Make NeoSound Database")
        self.Bind(wx.EVT_BUTTON, self.ClickMakeNeosound,self.makeneosoundbutton)
        grid.Add(self.makeneosoundbutton, (6,0), (1,1), wx.EXPAND)       

        # Import it! Pops up a window of summary stats of HDF5 file and asks
        # you to verify import, then does it
        self.importbutton =wx.Button(self, label="Start Import!")
        self.Bind(wx.EVT_BUTTON, self.StartImport,self.importbutton)
        grid.Add(self.importbutton, (8,0), (1,1), wx.EXPAND)       
        
        
        self.SetSizerAndFit(grid)
        self.Show(True)
        # add a spacer to the sizer
        grid.Add((10, 40), pos=(9,0))



    def ClickRHDFile(self,event):
        print "One: ClickRHDFile",self.defaultrhddir
        fd = wx.FileDialog(self,style=wx.FD_OPEN)#defaultDir=self.defaultrhddir
        fd.SetDirectory(self.defaultrhddir)
        fd.ShowModal()
        self.rhdfile = fd.GetPath()
        print "Two:ClickRHDFile...",self.rhdfile
        self.rhdlabel.SetLabel(self.rhdfile)
        if len(self.savedir)>0:
            self.stimsavefilename = os.path.join(self.savedir, os.path.basename(self.rhdfile).replace(".rhd", "_stimuli.h5"))
            self.stimsavefilenamelabel.SetLabel(self.stimsavefilename)
        
    def ClickStimFile(self,event):
        print "Open"
        fd = wx.FileDialog(self,style=wx.FD_OPEN)
        if len(self.rhdfile)>0:
            self.defaultcsvdir = os.path.basename(self.rhdfile)
        fd.SetDirectory(self.defaultcsvdir)
        fd.ShowModal()
        self.stimfile = fd.GetPath()
        print "On Open...",self.stimfile
        self.stimfilelabel.SetLabel(self.stimfile)        
        # import pyoperant generated stim file (.csv)
        self.csv_reader = stim.StimCSVReader()       
        print "log file of stimuli looks something  like: ", self.csv_reader.stim_data[0:10].astype(str)             
        
    def ClickStimDir(self,event):
        dd = wx.DirDialog(self,style=wx.DD_DEFAULT_STYLE)
        dd.ShowModal()
        self.stimdir = dd.GetPath()
        self.stimdirlabel.SetLabel(self.stimdir)  

    def StimSaveFileNameButton(self, event):
        self.stimsavefilenamelabel.SetLabel(self.stimsavefilenametextctrl.GetValue())
     
    def GenStimSaveFileNameButton(self, event):
    #generate a filename for saving stimulus file based on the RHD file and the saved directory
        if len(self.rhdfile)<1:
            print "Select an RHD file"
            pop = wx.MessageDialog(self,message = "Select an RHD File first (for the filename...)", style = wx.OK)
            pop.ShowModal()
            #generate dialog saying error
        elif len(self.savedir)<1:
            print "Select a directory for saving to"
            pop = wx.MessageDialog(self,message = "Select a directory for saving to (for the filename...)", style = wx.OK)
            pop.ShowModal()
            #generate dialog saying error
        else:
            self.stimsavefilename = os.path.join(self.savedir, os.path.basename(self.rhdfile).replace(".rhd", "_stimuli.h5"))
            self.stimsavefilenamelabel.SetLabel(self.stimsavefilename)
            self.stimsavefilenametextctrl.SetValue(self.stimsavefilename)

    def ClickSaveDir(self,event):
        print "Open"
        dd = wx.DirDialog(self,style=wx.DD_DEFAULT_STYLE)
        dd.ShowModal()
        self.savedir = dd.GetPath()
        print "On Open...",self.savedir
        self.savedirlabel.SetLabel(self.savedir)         
        
        
        
    def ClickMakeNeosound(self,event):
        # Read all of the metadata and trial data        
        #The stimulus data is first read out using the class `StimCSVReader`. This basically: 
        #1. reads the pyoperant csv file
        #2. keeps the trial and stimulus filename columns
        #3. modifies each stimulus filename to point to a different directory (the `root_dir` argument)
        #4. parses the last part of the filename for metadata according to Julie's naming conventions.
        #5. adds this metadata to the table as well
        #6. loads each stim from disk and writes it out into a neosound database with all metadata added as an annotation on the sound.
        print "ClickMakeNeosound def not finished..."
        try:
            assert len(self.rhdfile)>0
        except AssertionError: 
            print "Select an RHD file"
            pop = wx.MessageDialog(self,message = "Select an RHD file first (for the filename...)", style = wx.OK)
            pop.ShowModal()

        stimulus_h5_filename = os.path.join(self.stimdir,
                        os.path.basename(self.rhdfile).replace(".rhd", "_stimuli.h5"))
        print "self.rhdfile = ", self.rhdfile
        print "os.path.basename(self.rhdfile) ",os.path.basename(self.rhdfile)
        print "stimulus_h5_filename ",stimulus_h5_filename
        
        self.csv_reader = stim.StimCSVReader()
        
        #   Read all of the metadata and trial data
        self.csv_reader.read(self.stimfile)#, root_dir=self.stimdir)

        # Write sounds to a neosound database
    #    DONT DO UNTIl program works, but then DO THIS
        #self.csv_reader.write(stimulus_h5_filename)

        #display stimtagdisplay (example of stimulus tags) - this is a pain and i'm stopping
#        str_df = self.csv_reader.stim_data[0:10].astype(str) #dataframe in string format
        stim_str = self.csv_reader.stim_data[0:10].to_string
        str_df = self.csv_reader.stim_data.astype(str)
        print stim_str, type(stim_str)
        print str_df, type(str_df)
        self.stimtagdisplay.SetValue(self.csv_reader.stim_data[0:10].to_string)
#        lambda x: list(map(sys.stdout.write,x))
#        

#         
############FROM INTERWEBS
#    def get_lines_fast_struct(self,col_space=1):
#        """ lighter version of pandas.DataFrame.to_string()
#            with special spacing format"""
#        df_recs = self.to_records(index=False) # convert dataframe to array of records
#        str_data = map(lambda rec: map(str,rec), df_recs ) # map each element to string
#        self.space = map(lambda x:len(max(x,key=len))+col_space,  # returns the max string length in each column as a list
#                         zip(*str_data)) 
#
#        col_titles = [self._format_line(list(self))]
#        col_data = [self._format_line(row) for row in str_data ]
#
#        lines = col_titles + col_data
#        return lines
#######################################         
#         
#         
##        
##        print "str_df",type(str_df), len(str_df), sr_df
#        
##        self.stimtagdisplay.SetValue(str_df)
##        stimuli = self.csv_reader.stim_data.to_string()
#        str_df = self.csv_reader.stim_data.astype(str)
#        # I'm struggling to actually make this read in the window so I'm going to stop trying
#        
#        col_titles, col_data = get_lines_fast_struct2(csv_reader.stim_data)      
#        

#        self.stimtagdisplay.SetValue(self.csv_reader.stim_data[0:10].to_string())
        
        print "RHD file: ",self.rhdfile,"Stim tags file: ",self.stimfile,"save dir: ",self.savedir,"log file: ",self.logfile
        

    def StartImport(self,event):
        # THIS IS NOT FINISHED.....
        return

    def _format_line2(self, row_vals):
        return "".join(cell.rjust(width) for (cell, width) in zip(row_vals, self.space))
        
    def get_lines_fast_struct2(self, col_space=1):
        str_df = self.astype(str)
        self.space = [str_df[c].map(len).max() for c in str_df.columns]
        col_titles = map(_format_line2, [self.columns])
        col_data = map(_format_line2, str_df.to_records(index=False))
        return col_titles + col_data 


#
#    def EvtText(self, event):
#        self.logger.AppendText('EvtText: %s\n' % event.GetString())
#    def EvtChar(self, event):
#        self.logger.AppendText('EvtChar: %d\n' % event.GetKeyCode())
#        event.Skip()
#    def EvtCheckBox(self, event):
#        self.logger.AppendText('EvtCheckBox: %d\n' % event.Checked())

if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(None,-1,'RHD Importer', size=(1000,500))
    panel = RHDImporter(frame)
    frame.Show()
    app.MainLoop()