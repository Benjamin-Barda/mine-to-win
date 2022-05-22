class _proposal : 


    def __init__(self) : 
        pass


    def __call__(self, fg_scores, reg_scores, anchors, img_size) :
        '''
        args : 
            fg_score
            reg_scores
            anchors
            img_size
        return : 
            RoI
        
        Alg : 
            Convert Anchors into proposal ... that is apply offsets from reg to anchor and clip it to image
            Sort them based on fg_scores
            Apply NMS and take only top K
        ''' 
        pass
        


