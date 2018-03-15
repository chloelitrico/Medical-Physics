import os
import numpy as np
from skimage.draw import polygon
import dicom

    
def get_ROI_Contour_Points(ROI_Name, RS_filepath):
    """
    Function: extracts the contour points of a specific ROI from an RS file
    
    Input:
    ROI_Name: name of the region of interest found in the specified RS file 
    RS_filepath: path from current directory to region where RS.dcm file is located
    
    Output:
    A 3D array of the contour points of the ROI
    """
    
    rs = dicom.read_file(RS_filepath)
    
    #Find the index at which the ROI is located in the RS file
    for index, item in enumerate(rs.StructureSetROISequence):
        if item.ROIName == ROI_Name:
            ROI_index = index
    
    #Get all contour points from RS file (organized as: [[x0-N, y0-N, z0-N][x0-N, y0-N, z0-N]] )
    contour = []
    for item in rs.ROIContourSequence[ROI_index].ContourSequence:
        contour.append(item.ContourData)
    return np.array(contour)


def convert_mm_to_index(x_vals, y_vals, image_position, pixel_spacing):
    """
    Function: converts mm measurements of ROI contour points to pixel index values
    
    Input:
    x_vals: x values of ROI contour points of one slice
    y_vals: y values of ROI contour points of one slice
    ***NOTE: x and y values are to be in order with their point
    image_position: top left corner of the CT image
    pixel_spacing: mm size of one pixel in CT image
    
    Output:
    A list of x and y pixel index values 
    """
    
    #Get pixel index values
    x_val_pixel_index = [abs(int(round((image_position[0] - x)/pixel_spacing[0], 0))) for x in x_vals]
    y_val_pixel_index = [abs(int(round((image_position[1] - y)/pixel_spacing[1], 0))) for y in y_vals]
    return x_val_pixel_index, y_val_pixel_index
    
def mask_per_slice(slice_contour, image_position, image_shape, pixel_spacing):
    """
    Function: Makes a mask -- value: 1 inside the contour point and 0 outside contour point for each slice

    Input: 
    slice_contour: contour points of one slice
    image_position: top left corner of the CT image
    image_shape: number of pixel rows and columns of the CT image (? X ?)
    pixel_spacing: mm size of one pixel in CT image
    """
    
    #Extract X contour values of one slice 
    x_vals = slice_contour[0::3]
    #Extract Y contour values of one slice 
    y_vals = slice_contour[1::3]
    #Extract Z value of one slice 
    slice_number = slice_contour[2] 
    
    x, y = convert_mm_to_index(x_vals, y_vals, image_position, pixel_spacing)
    
    #Make an empty 2D array of 0z -- with the same size as our CT image (same amount of pixel columns and rows)
    matrix = np.zeros(image_shape, dtype=np.uint8)
    #Row coordinates of vertices of polygon
    r = np.array(y)
    #Column coordinates of vertices of polygon
    c = np.array(x)
    rr, cc = polygon(r, c)
    #fill area of matrix found within volume with 1s
    matrix[rr, cc] = 1

    return slice_number, matrix
 
    
def get_mask(ROI_Name, RS_filepath, CT_folderpath):
    """
    Function: makes a mask with 1: inside volume, 0: outside volume depending on contour points of an ROI
    
    Input:
    ROI_Name: name of the region of interest found within the RS file
    RS_filepath: filepath to the RS folder
    CT_folderpath: path to CT folder containing all CT images of each slice
    
    Output: 
    mask: matrix containing 1s (region inside ROI contour) and 0s (outside ROI contour)
    slice_numbers: z-value of slices in order of the 3D mask 
    """
   
    #Obtain a list of the files found within the CT folder
    CT_slices = os.listdir(CT_folderpath)
    
    #Assuming all CT images in the same folder are of the same shape and position
    ds = dicom.read_file(CT_folderpath + '/' + CT_slices[0])
    
    slice_numbers = []
    mask = []
    #Get number of pixel rows and columns of the CT image
    image_shape = (ds.Rows, ds.Columns)
    #Get top left corner of CT image
    image_position = [float(x) for x in ds.ImagePositionPatient]
    pixel_spacing = [float(x) for x in ds.PixelSpacing]

    #Get matrix for each slice and create 3D matrix
    for slice_contour in get_ROI_Contour_Points(ROI_Name, RS_filepath):
        slice_n, m = mask_per_slice(slice_contour, image_position, image_shape, pixel_spacing)
        slice_numbers.append(slice_n)
        mask.append(m)
   
    #To check if slices are in order
    if slice_numbers != sorted(slice_numbers):
        slice_numbers, mask = (list(t) for t in zip(*sorted(zip(slice_numbers, mask))))      
    
    return np.array(mask), slice_numbers


def get_hu(CT_folderpath, mask, slice_numbers):
    """
    Function: returns a list of the hounsfield units found within the mask
    
    Input:
    CT_folderpath: path to CT folder containing all CT images of each slice
    mask: 3D matrix contianing 1s (volume inside ROI), and 0 (volume outside ROI)
    slice_numbers: the z value of slice found within the ROI (the z values of each slice of the mask)
    
    Output: 
    ROI_hu_vals: list of all hounsfield units found within the ROI
    """
    
    #list of names of all CT files in the folder containing CT files of one CT scan
    CT_files = os.listdir(CT_folderpath)

    slice_location = []
    CT_slices = []

    #only save CT_file slices that are found within the mask
    for CT_file in CT_files:
        ds = dicom.read_file(CT_folderpath + '/' + CT_file) 
        if ds.SliceLocation in slice_numbers:
            slice_location.append(ds.SliceLocation)
            CT_slices.append(CT_file)

    #To check if slices are in order -- puts them in order if they are not
    if slice_location != sorted(slice_location):
        slice_location, CT_slices = (list(t) for t in zip(*sorted(zip(slice_location, CT_slices))))
    
    #Open CT files that are found wihtin the mask -- convert pixel values to hounsfield units
    ROI_hu_vals = []
    for i, contour_slice in enumerate(mask):
        ds = dicom.read_file(CT_folderpath + '/' + CT_slices[i])
        pixel_image = ds.pixel_array
        slope = ds.RescaleSlope
        intercept = ds.RescaleIntercept
        hu_image = pixel_image * slope + intercept
        
        #go through each value in the slice, and save values which are found within the ROI
        for i, row in enumerate(contour_slice):
            for j, voxel in enumerate(row):
                if voxel == 1:
                    ROI_hu_vals.append(hu_image[i][j])
    return ROI_hu_vals


def get_HU_txt(RS_filepath, ROI_Name, CT_folderpath, output_folderpath, patient_x):
    """
    Function: returns a list of the hounsfield units in a txt file
    
    Input:
    RS_filepath: filepath to the RS folder
    ROI_Name: name of the region of interest found within the RS file
    CT_folderpath: path to CT folder containing all CT images of each slice
    output_folderpath: path to specified folder/creates folder that will contain all HU.txt files
    patient_x: patient number -- to identify from other patients 
    
    Output: 
    A txt file with HU vals
    """

    #Get mask and slice_numbers
    mask, slice_numbers = get_mask(ROI_Name, RS_filepath, CT_folderpath)
    #Get HU
    hounsfield_units = get_hu(CT_folderpath, mask, slice_numbers)
    
    #Assuming all CT images in the same folder are of the same shape and position    
    ct = dicom.read_file(CT_folderpath + '/' + os.listdir(CT_folderpath)[0])
    
    #check if output folder exists
    directory = os.path.dirname(output_folderpath + "/" + patient_x + "_" + ct.StudyDate + ".txt")
    #If the output folder does not exist, create folder
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    #write HUs to txt file
    f = open(output_folderpath + "/" + patient_x + "_" + ct.StudyDate + ".txt",'w')
    f.write("Hounsfield Units: \n" + ', '.join(str(x) for x in hounsfield_units)+ "\n")
    
    #Save certain values to txt file
    f.write("Patient's Birth Year: \n" + ct.PatientsBirthDate[0:3] + "\n")
    f.write("Voxel Diameter (x, y, z): \n"  + ", ".join(str(x) for x in ct.PixelSpacing) + ", " + str(ct.SliceThickness) + "\n")
    f.close()