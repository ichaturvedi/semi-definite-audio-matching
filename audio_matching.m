function match = audio_matching(datasetFolder,goldAudio)

[audioIn,fs] = audioread(goldAudio);

f = phoneme(audioIn(1:10,1));
for i=1:size(audioIn,1)/10-1
    f = [f phoneme(audioIn(i*10+1:(i+1)*10,1))];
end


audioInm = mfcc(audioIn(:,1),fs);
gm = fitgmdist(audioInm,8,'Regularize',1e-5);
audioIn = audioIn(:,1);

y = audioIn(1:end-4)';
x = [audioIn(2:end-3)';audioIn(3:end-2)';audioIn(4:end-1)'];
e1 = lsqr(x',y');
e2 = cov(x');

q = polyint((e1.^2)');
a = 0;
b = 2; % max time delay
I = diff(polyval(q,[a b]));

R = sdpvar(3,3,'symmetric'); % for Lyapunov
D = sdpvar(1,3,'full');
C1 = [D>0];
C2 = [e1'*R*e1>=I];
C3 = [(e1'*x)*(e1'*x)'*D*D'>(R*e2*R'*e2*x*x'*D')'*D'];
O = [C1,C2,C3];
solvesdp(O); 
checkset(O);

R_gold = R;
D_gold = D;
close all

filenames  = getAllFiles(datasetFolder);

parfor ii=1:length(filenames)

[audioIn,fs] = audioread(filenames{ii});
audioIn = mfcc(audioIn(:,1),fs);
audioIn = audioIn(:,1);

y = audioIn(1:end-4)';
x = [audioIn(2:end-3)';audioIn(3:end-2)';audioIn(4:end-1)'];
e1 = lsqr(x',y');
e2 = cov(x');

q = polyint((e1.^2)');
a = 0;
b = 2;
I = diff(polyval(q,[a b]));

R = sdpvar(3,3,'symmetric'); % for Lyapunov
D = sdpvar(1,3,'full');
C1 = [D>0];
C2 = [e1'*R*e1>=I];
C3 = [(e1'*x)*(e1'*x)'*D*D'>(R*e2*R'*e2*x*x'*D')'*D'];
O = [C1,C2,C3];
solvesdp(O); 
checkset(O);

match(ii) = mse(double(R),R_gold)+mse(double(D),D_gold);
close all

end

dlmwrite('matching_error.txt',match');
end
