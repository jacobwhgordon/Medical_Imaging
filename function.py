#Functions and tools for pydicom image processing/this project spicifically.

import numpy as np
import matplotlib.pyplot as plt

PATH = 'E:\\DicomData\\LUNG1\\NSCLC-Radiomics\\'

# Stuff for dicom data
import os
import pydicom
from pydicom.data import get_testdata_files


from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from skimage.draw import polygon
from skimage.draw import circle
from scipy import ndimage

#Function to load and sort a set of slices
def loadScan(folder):
    slices = [pydicom.dcmread(folder+filename) for filename in os.listdir(folder)]
    #slices.sort(key = lambda x: int(x.InstanceNumber))   #for a few samples, instance number doesnt work. weird...
    slices.sort(key = lambda x: int(-x.SliceLocation)) 
    return slices

#Function to load a contour, contours dont need sorting.
def loadContour(folder):
    con = [pydicom.dcmread(folder+filename) for filename in os.listdir(folder)]
    return con

#Function to load all scans and contours for a given patient.
def loadPatient(patient):
    data = []
    for s in os.listdir(PATH+patient+"\\"):
        dir2 = [d for d in os.listdir(PATH+patient+"\\"+s+"\\")]        
        #print("  " + s)
        folder = ""
        folderC = ""
        if len(dir2) > 1:
            #We need to check which folder has the slices vs the contours.  There is a bunch of slices, but 1 contour.
            if (len(os.listdir(PATH+patient+"\\"+s+"\\"+dir2[1]))) == 1:
                #print("    " + dir2[0])
                #print("    " + dir2[1])
                folder = PATH+patient+"\\"+s+"\\"+dir2[0]+"\\"
                folderC = PATH+patient+"\\"+s+"\\"+dir2[1]+"\\"
            else:
                #print("    " + dir2[1])
                #print("    " + dir2[0])
                folder = PATH+patient+"\\"+s+"\\"+dir2[1]+"\\"
                folderC = PATH+patient+"\\"+s+"\\"+dir2[0]+"\\"
            contour = loadContour(folderC)
        else: 
            #print("    " + dir2[0])
            folder = PATH+patient+"\\"+s+"\\"+dir2[0]+"\\"
            contour = []
        slices = loadScan(folder)
        data=[slices,contour]
    return data

#Function to plot a sub set of the slices in a stack
def plotSlices(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[2*cols,2*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
        ax[int(i/cols),int(i % cols)].imshow(stack[ind].pixel_array, cmap=plt.cm.bone)
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()
    
#Function to plot a sub set of slices if our stack is already pixel maps.
def plotPixelStack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[2*cols,2*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
        ax[int(i/cols),int(i % cols)].imshow(stack[ind], cmap=plt.cm.bone)
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()
    
def plotMask(image,mask_image,label, save = False, filename = 'img.png'):
    fig,ax = plt.subplots(1,3,figsize=[12,4])
    ax[0].set_title('Original Scan')
    ax[0].imshow(image, cmap = plt.cm.bone)
    ax[0].axis('off')
    ax[1].set_title('Original Scan with label')
    ax[1].imshow(image, cmap = plt.cm.bone)
    ax[1].imshow(label, alpha = 0.3)
    ax[1].axis('off')
    ax[2].set_title('Masked Scan with label')
    ax[2].imshow(mask_image, cmap = plt.cm.bone)
    ax[2].imshow(label, alpha = 0.3)
    ax[2].axis('off')
    '''
    N=len(image)
    ax[1,1].set_title('50% Zoomed Scan with label')
    ax[1,1].imshow(image[int(0.25*N):int(0.75*N),int(0.25*N):int(0.75*N)], cmap = plt.cm.bone)
    ax[1,1].imshow(label[int(0.25*N):int(0.75*N),int(0.25*N):int(0.75*N)], alpha = 0.3)
    ax[1,1].axis('off')
    '''
    plt.show()    
    
    if save:
        fig.savefig(filename)

# Function to mask out everything that is not the lungs.
#
# How will we do this:
# - First we will standadize the pixel values around 0 by subtracting the mean, and dividing by the standard deviation.
# - Next we will use a k-mean algorythm to seperate the pixels into two populations (soft tissue/bone vs lung/air)
# - Then we will use errosion and dialation to remove some of the smaller features so its easier to find
#   the boundries of the lungs
# - Identify and lable the seperate regions.
# - Seperate the regions into 'Lung' and 'everything else'
# - Create a mask for the lung region.
# - Apply the mask to the ogiginal slice to get rid of everything but the lungs.

# pass the function a pixle map, pmap

def maskLung(pmap, verbose = False):
    # get mean and std and normalize pixels, and some other image properties.
    num_rows = pmap.shape[0]
    num_cols = pmap.shape[1]
    mean = np.mean(pmap)
    std = np.std(pmap)
    
    #normalize
    
    #Pixel values for these scans vary between 0 and 4095 (note, for output we want 0-255)
    #and some which vary between -1024 to 3071 (idk why they are different?)
    #the important thing to note is not all full that whole range!
    #for now normalize so max range is 0 to 1.
    pmap = pmap - np.min(pmap)
    pmap = pmap / 4095.
    
    #Old normalization
    #pmap = (pmap - np.min(pmap))/(np.max(pmap)-np.min(pmap))  
    
    #Very old normalization
    #pmap = pmap-mean
    #pmap = pmap/std
    
    if verbose:
        print("["+str(num_cols)+","+str(num_rows)+"] pixel array with mean at "+str(mean)+" and std at "+str(std))
        print("")
        print("Image pre-processing")
        plt.imshow(pmap, cmap=plt.cm.bone)
        plt.show()
    
    # get the middle-ish area (near the lung) (we cut out the outside 20%) as a starting guess for our kmean boundry.
    middle = pmap[int(num_cols*0.2):int(num_cols*0.8),int(num_rows*0.2):int(num_rows*0.8)]
    
    # if we had underflow and overflow in our pixel maps, we would want to move those onto our normalized scale now,
    # but as far as I can tell, we dont for these maps.
    
    # Use KMeans to seperate lungs/air vs soft tissue/bone.
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())                             #Identify the centers of the 2 cluster
    threshold = np.mean(centers)                                                    #Define the threshold
    thresh_pmap = np.where(pmap<threshold,1.0,0.0)                                  #Set below thresh to 1.0, and above to 0.0
    
    if verbose:
        print("Centers from KMeans: "+str(centers)+" \nThreshold set at: "+str(threshold))
        print("")
        print("Image post threshold processing:")
        plt.imshow(thresh_pmap, cmap=plt.cm.bone)
        plt.show()
        
    # Now we erode and diolate to remove very small fetures and noise, and diolate so we 
    # include a few pixels around the lung to prevent us from cutting parts of the lung.
    
    eroded = morphology.erosion(thresh_pmap,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))
    
    if verbose:
        #print("Centers from KMeans: "+str(centers)+" \nThreshold set at: "+str(threshold))
        #print("")
        print("Image post erosion and dilation:")
        plt.imshow(dilation, cmap=plt.cm.bone)
        plt.show()
        
    # Now we want to label the different areas
    labels = measure.label(dilation)              # Each connected area is given its own value
    label_vals = np.unique(labels)                # Returns a list of unique values from labels
    regions = measure.regionprops(labels)         # Measures the properties of the labeled regions
    
    if verbose:
        print("Color map of different regions:")
        plt.imshow(labels) #, cmap=plt.cm.bone)
        plt.show()
        
    
    # Lets figure out which labels go with the lungs
    good_labels = []
    for properties in regions:
        B = properties.bbox          #Returns (y_min,x_min,y_max,x_max)
        min_row = B[0]
        max_row = B[2]
        min_col = B[1]
        max_col = B[3]
        if  ( (max_row-min_row) < (0.9*num_rows) and     # if the rows span less than 90%
              (max_col-min_col) < (0.9*num_cols) and     # and the columns span less than 90%
              (min_row) > (0.2*num_rows) and             # and the rows are inside the middle 60% of the frame
              (max_row) < (0.8*num_rows)
            ):
            good_labels.append(properties.label)         # Then this region must be a good one!
            
            
    # Now we create the mask
    # We also diolate it once more with a larger size, to smooth/fill in the lung mask.
    mask = np.ndarray([num_rows,num_cols],dtype=np.int8)
    mask[:] = 0
    for label in good_labels:
        mask = mask + np.where(labels==label,1,0)
    mask = morphology.dilation(mask,np.ones([10,10]))
    
    if verbose:
        print("Final Lung Mask:")
        plt.imshow(mask, cmap=plt.cm.bone)
        plt.show()
    
    
    # Finally, apply the mask (and output if verbose), renormalize, and set the areas that are cut and set to zero to
    # -mean/std, because they should be at the minimum, not the middle.
    
    new_pmap = pmap*mask
    #if np.mean(mask) != 0:
    #    
    #    new_mean = np.mean(new_pmap[np.nonzero(mask)])  #<- is this working?
    #    new_std = np.std(new_pmap[np.nonzero(mask)])
    #
    #    new_pmap = new_pmap-new_mean
    #    new_pmap = new_pmap/new_std
    
    
    if verbose:
        print("Final image with mask applied:")
        plt.imshow(new_pmap, cmap=plt.cm.bone)
        plt.show()
    
    return new_pmap
    
    
    
# Function to match contours to slices and output them  
# To do this we will need to
# - First get the individual contours and slices matched up
# - Next, get them into the same coordinate system 
# - Then third convert into label map

def labelTumor (patient, verbose = False):
    
    slices = patient[0]
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    
    if (patient[1] == []):
        print("No contours manually identified")
        return np.zeros_like(image, dtype=np.uint8),tuple(np.array([]))
    contours = patient[1][0]
    
    #Define the cordinate system
    
    z = [s.ImagePositionPatient[2] for s in slices]      # array of z positions
    pos_row = slices[0].ImagePositionPatient[1]          # row origin
    spacing_row = slices[0].PixelSpacing[1]              # row spacing
    pos_col = slices[0].ImagePositionPatient[0]          # column origin
    spacing_col = slices[0].PixelSpacing[0]              # column spacing
    
    
    
    label = np.zeros_like(image, dtype=np.uint8)

    #loop through the sets of contours that represent a tumor
    for i,contour in enumerate(contours.ROIContourSequence):
        
        #Save a lable for the tumor, incase there is more than one.        
        num = contour.ReferencedROINumber
        assert num == contours.StructureSetROISequence[i].ROINumber 
        
        #loop through the contours that are slices of a single tumor
        for con in contour.ContourSequence:
            
            nodes = np.array(con.ContourData).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            row = (nodes[:, 1] - pos_row) / spacing_row
            col = (nodes[:, 0] - pos_col) / spacing_col
            rr, cc = polygon(row, col)
            label[rr, cc, z_index] = num
               
    colors = tuple(np.array([con.ROIDisplayColor for con in contours.ROIContourSequence]) / 255.0)
    return label, colors

# for simplicity, we center a circular tumor mask on the image with a radius large enough to encompus the whole pixel by pixel mask.
def labelSimpleTumor (patient, verbose = False):
    
    slices = patient[0]
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    
    if (patient[1] == []):
        print("No contours manually identified")
        return np.zeros_like(image, dtype=np.uint8),tuple(np.array([]))
    contours = patient[1][0]
    
    #Define the cordinate system
    
    z = [s.ImagePositionPatient[2] for s in slices]      # array of z positions
    pos_row = slices[0].ImagePositionPatient[1]          # row origin
    spacing_row = slices[0].PixelSpacing[1]              # row spacing
    pos_col = slices[0].ImagePositionPatient[0]          # column origin
    spacing_col = slices[0].PixelSpacing[0]              # column spacing
    
    
    
    label = np.zeros_like(image, dtype=np.uint8)

    #loop through the sets of contours that represent a tumor
    for i,contour in enumerate(contours.ROIContourSequence):
        
        #Save a lable for the tumor, incase there is more than one.        
        num = contour.ReferencedROINumber
        assert num == contours.StructureSetROISequence[i].ROINumber 
        
        #loop through the contours that are slices of a single tumor
        for con in contour.ContourSequence:
            
            #double check to make sure we have a scan for this tumor slice (because apparently sometimes we don't...)
            if con.ContourData[2] not in z:
                continue
            
            nodes = np.array(con.ContourData).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            row = (nodes[:, 1] - pos_row) / spacing_row
            col = (nodes[:, 0] - pos_col) / spacing_col
            rr, cc = polygon(row, col)
            label[rr, cc, z_index] = num
            
            #plt.imshow(label[:,:,z_index])
            #plt.show()
            #Fill it out to be a simple circle
            center = ndimage.measurements.center_of_mass(label[:,:,z_index])
            radius = 0 
            for rrr in range(0,len(label[:,:,z_index])):
                for ccc in range(0,len(label[0,:,z_index])):
                    if label[rrr,ccc,z_index] != 0:
                        radius = np.max([radius,np.linalg.norm(center-np.array([rrr,ccc]))]) 
            
            #print(center[0],center[1],radius)
            if np.isnan(center[0]) or np.isnan(center[1]):
                continue
            rr, cc = circle(int(center[0]), int(center[1]), int(radius), shape = [512,512])
            label[rr, cc, z_index] = num
                             
                             
               
    colors = tuple(np.array([con.ROIDisplayColor for con in contours.ROIContourSequence]) / 255.0)
    return label, colors