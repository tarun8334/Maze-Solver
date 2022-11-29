# function to broaden the path we got fronm the path_find function
def sol_overlay(org_image,path,preprocess_image):
    image = org_image.copy()
    h = image.shape[0]
    w = image.shape[1]
    for i in range(1,len(path)):    

        # poition of previous point 
        prev_x = int(path[i-1]/w)
        prev_y = int(path[i-1]%w)
        # position of current point
        curr_x = int(path[i]/w)
        curr_y = int(path[i]%w)
        # checking if previous point and current point are in the same row
        if(abs(prev_x-curr_x)==1):
            # loop to broaden the path
            for y in range(curr_y, curr_y-5,-1):
                # breaking when we encounter a maze wall point and similar thing done below as well
                if(y<0 or preprocess_image[curr_x][y]==255):
                    break
                image[curr_x][y] = (255,0,0)
                
            for y in range(curr_y, curr_y+5):
                if(y>=w or preprocess_image[curr_x][y]==255):
                    break
                image[curr_x][y] = (255,0,0)
        
        else:
            for x in range(curr_x, curr_x-5,-1):
                if(x<0 or preprocess_image[x][curr_y]==255):
                    break
                image[x][curr_y] = (255,0,0)
                
            for x in range(curr_x, curr_x+5):
                if(x>=h or preprocess_image[x][curr_y]==255):
                    break
                image[x][curr_y] = (255,0,0)

    return image