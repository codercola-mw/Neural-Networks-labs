function [H] = StrongClassifier(xTest, am,tm,pm,idm)

   n = size(xTest,2);
   h = zeros(length(am), n);
   
   for i = 1:length(am)
       h(i,:)=am(i).*WeakClassifier(tm(i),pm(i),xTest(idm(i),:));
  
       hs = signed( sum(h(1:i,:),1);
   end
   H = sign(sum(h,1));

end
 