Region Proposal NETWORK: 
    IN -> ultima feature map della backbone

    CONV (in.shape -> dimensione fissa : 256 esempio) + RELU 
    questa viene feedata un due altri layer : 
        Regression : 256 -> 4 * numero ancore
        Classification : 256 -> 2 * numero ancore
    per ogni N = Batch size: 
        Prendiamo OUT di Regression e classification + Anchore e le buttiamo nel region proposal LAYER (dove avviene la magia): 


REGION PROPOSAL LAYER: 
    Convertiamo ancore applicando offset del regressore. 
    clippiamo ancore all'immagine originale
    leviamo ancore che sono fuori dal bordo o troppo piccole
    Sortiamo ancore in base a score fg/bg e facciamo NMS (non max suppression)
    e queste sono le nostre ROI
        calcoliamo IoU sulle ROI e ne teniamo un X



    tornando a rpn ritorna con qualche altra info ROI, e gli score di regressore e classificatore


Alla fine si passa tutto in altre CONV e poi si risplitta> 
    classificazione oggetto 
    rifinizione BBox



    

