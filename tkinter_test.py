# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:28:13 2017

@author: billewood
"""

from Tkinter import Tk, Text, BOTH, W, N, E, S, Listbox
from ttk import Button, Label, Style


class ImportGUI(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        self.initUI()
        
        
    def initUI(self):
      
        self.parent.title("Import RHD to Neo")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)
        
        lbl = Label(self, text="Windows")
        lbl.grid(sticky=W, pady=4, padx=5)
        
        area = Text(self)
        area.grid(row=1, column=0, columnspan=2, rowspan=4, 
            padx=5, sticky=E+W+S+N)
        
        abtn = Button(self, text="Activate")
        abtn.grid(row=1, column=3)

        cbtn = Button(self, text="Close")
        cbtn.grid(row=2, column=3, pady=4)
        
        hbtn = Button(self, text="Help")
        hbtn.grid(row=5, column=0, padx=5)

        obtn = Button(self, text="OK")
        obtn.grid(row=5, column=3)        
#        
#        lb = Listbox(self)
#        epoch = 0
#        for i in epoch:
#            lb.insert(END, i)
#            
#        lb.bind("<<ListboxSelect>>", self.onSelect)    
#            
#        lb.pack(pady=15)
        
    def onSelect(self, val):
      
        sender = val.widget
        idx = sender.curselection()
        value = sender.get(idx)   

        self.var.set(value)
#    def displayPlaybackCodes(self):

        
        
#class RHDfile(self):
    
              

def main():  
    root = Tk()
    root.geometry("700x600+600+600")
    app = ImportGUI(root)
    root.mainloop()  


if __name__ == '__main__':
    main()  