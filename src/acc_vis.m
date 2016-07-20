


for L3_NUM = [2,4,8,16,32,64,128,256,512,1024]
    x = L1(L3==L3_NUM);
    y = L2(L3==L3_NUM);
    z = testacc(L3==L3_NUM);
    
    xlin = linspace(min(x),max(x),33);
    ylin = linspace(min(y),max(y),33);
    
    [X,Y] = meshgrid(xlin,ylin);
    
    f = scatteredInterpolant(x,y,z);
    Z = f(X,Y);
    
    
    surf(X,Y,Z)
    title(['Fully Connected Size : ',num2str(L3_NUM)]);
    xlabel('Layer 1 Size');
    ylabel('Layer 2 Size');
    zlabel('Accuracy');
    grid on
    pause
    clf
end