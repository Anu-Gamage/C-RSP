function sw = swiss_roll(m)
    plot = 0;
    noise_power = 0;
    n = 1000;
    a = 1;
    b = 1;
    num_spirals = 1.5;
    x1 = .4;
    y1 = .3;
    r1 = .2;
    x2 = .7;
    y2 = .4;
    r2 = .1;

    points = rand(n,2).*[a b];
    while 1
        circles = (points(:,1)-x1).^2 + (points(:,2)-y1).^2 < r1^2 ...
                | (points(:,1)-x2).^2 + (points(:,2)-y2).^2 < r2^2;
        if any(circles)
            points(circles,:) = rand(sum(circles),2).*[a b];
        else
            break
        end
    end
    points = sortrows(points);
    
    points(:,3) = points(:,1).*sin(num_spirals/a*2*pi*points(:,1));
    points(:,1) = points(:,1).*cos(num_spirals/a*2*pi*points(:,1));
    
    if noise_power
        points = points + noise_power*randn(size(points));
    end
    
    if plot
        t = (1/1000:1/1000:1)';
        circle1 = -[r1*cos(2*pi*t)-x1, r1*sin(2*pi*t)-y1; r2*cos(2*pi*t)-x2, r2*sin(2*pi*t)-y2];
        circle1(circle1(:,1) < 0 | circle1(:,1) > a | circle1(:,2) < 0 | circle1(:,2) > b,:) = [];

        circle1(:,3) = circle1(:,1).*sin(num_spirals/a*2*pi*circle1(:,1));
        circle1(:,1) = circle1(:,1).*cos(num_spirals/a*2*pi*circle1(:,1));

        figure(1); clf;
        scatter3(points(:,1),points(:,2),points(:,3),[],sqrt(points(:,1).^2 + points(:,3).^2)/a);
        hold on;
        scatter3(circle1(:,1),circle1(:,2),circle1(:,3),'.');
    end
    
    % x = pitch (n/a)
    % y = yaw
    % z = roll
    sw = struct;
    angles = pi*rand(2,m);
%     angles = [0 pi/3 pi/2 2*pi/3];
%     anglestr = {'0','pi_3','pi_2','2pi_3'};
%     for ii = [2 4]
%         for jj = [1 4]
%             name = ['phi' anglestr{ii} 'th' anglestr{jj}];
%             phi = angles(ii);
%             th = angles(jj);
    for ii = 1:m
        name = ['view' num2str(ii)];
        phi = angles(1,ii);
        th = angles(2,ii);
        PhiM = [cos(phi) 0 -sin(phi);0 1 0;sin(phi) 0 cos(phi)];
        ThM = [cos(th) -sin(th) 0;sin(th) cos(th) 0;0 0 1];
        points_xyz = points*ThM*PhiM;

        if plot
            figure();
            scatter(points_xyz(:,1),points_xyz(:,2),[],sqrt(points(:,1).^2 + points(:,3).^2)/a);
        end

        A = pdist2(points_xyz(:,1:2),points_xyz(:,1:2));
        sigma = median(A(:));
        sw.A.(name) = exp(-A.^2/(2*sigma^2));
    end
%         end
%     end
    sw.gt = points;
end