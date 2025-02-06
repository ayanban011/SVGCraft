"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math

pydiffvg.set_print_timing(True)

gamma = 1.0

def main(args):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
    
    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    shapes = []
    shape_groups = []
    if args.use_blob:
        for i in range(num_paths):
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()]))
            shape_groups.append(path_group)
    else:
        start_y = 1
        end_y = 256
        current_patch = 0
        skip_patches=[]
        for a in range(8):
            start_x = 1
            end_x = 256

            for b in range(8):
                if current_patch in skip_patches:
                    current_patch = current_patch + 1
                    start_x = start_x + 256
                    end_x = end_x + 256
                    continue
                #circle
                x = random.randint(start_x, end_x)
                y = random.randint(start_y, end_y)
                max_radius_x = 32/2
                max_radius_y = 32/2
                if x < start_x + 32/2:
                    max_radius_x = x - start_x
                else:
                    max_radius_x = end_x - x
                if y < start_y + 32/2:
                   max_radius_y = y - start_y
                else:
                    max_radius_y = end_y - y

                max_radius = min(max_radius_x, max_radius_y)
                if max_radius == 0:
                    max_radius = 1

                r = random.randint(1,max_radius)

                path = pydiffvg.from_svg_path(f'M {x - r}, {y} a {r},{r} 0 1,1 {r*2},0 a {r},{r} 0 1,1 {-1 * r * 2},0') 
                for i in path:
                    shapes.append(i)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            stroke_color = torch.tensor([1.0,0.0,0.0,1.0]), #red
                                            fill_color = None)
                shape_groups.append(path_group)

                #Line
                x1 = random.randint(start_x, end_x)
                y1 = random.randint(start_y, end_y)
                x2 = random.randint(start_x, end_x)
                y2 = random.randint(start_y, end_y)
                path = pydiffvg.from_svg_path(f'M {x1},{y1} L {x2},{y2}') #line
                for i in path:
                    shapes.append(i)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                        stroke_color = torch.tensor([0.0,0.0,0.0,1.0]),#green
                                        fill_color = None)
                shape_groups.append(path_group)

                #square
                x = random.randint(start_x,end_x)
                y = random.randint(start_y,end_y)
                h = random.randint(start_x,end_x)
                v = y+h-x

                if v < 0 or v > 32:
                    v = y+x-h

                path = pydiffvg.from_svg_path(f'M {x} {y}, H {h}, V {v}, H {x}, V{y}') #square
                for i in path:
                    shapes.append(i)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            stroke_color = torch.tensor([0.8,0.6,0.4,1.0]), #orange
                                            fill_color = None)
                shape_groups.append(path_group)

                #triangle
                x1 = random.randint(start_x,end_x)
                y1 = random.randint(start_y,end_y)
                x2 = random.randint(start_x,end_x)
                y2 = random.randint(start_y,end_y)
                x3 = random.randint(start_x,end_x)
                y3 = random.randint(start_y,end_y)
                path = pydiffvg.from_svg_path(f'M {x1} {y1} L {x2} {y2} L       {x3} {y3} Z')
                for i in path:
                    shapes.append(i)
                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            stroke_color = torch.tensor([0.0,0.0,1.0,1.0]), #blue
                                            fill_color = None)
                shape_groups.append(path_group)

                #Semi-circles
                x = random.randint(start_x, end_x)
                y = random.randint(start_y, end_y)
                max_radius_x = 32/2
                max_radius_Y = 32/2
                if x < start_x + 32/2:
                   max_radius_x = x - start_x
                else:
                    max_radius_x = end_x - x
                if y < start_y + 32/2:
                    max_radius_y = y - start_y
                else:
                   max_radius_y = end_y - y
                
                max_radius = min(max_radius_x, max_radius_y)
                if max_radius == 0:
                    max_radius = 1

                r = random.randint(1, max_radius)
                path = pydiffvg.from_svg_path(f'M {x - r}, {y} a {r},{r} 0 1,1 {r*2},0')
                for i in path:
                    shapes.append(i)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor       ([len(shapes) - 1]),
                                                stroke_color = torch.tensor([0.5,0.0,0.5,1.0]), #purple
                                               fill_color = None)
                shape_groups.append(path_group)

                #L-shape
                x = random.randint(start_x,end_x)
                y = random.randint(start_y,end_y)
                v = random.randint(start_y,end_y)
                h = random.randint(start_x,end_x)
                path = pydiffvg.from_svg_path(f'M {x} {y} V {v} H {h}')
                for i in path:
                    shapes.append(i)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), 
                                            stroke_color = torch.tensor([1.0,1.0,0.0,1.0]), #yellow
                                           fill_color = None)

                shape_groups.append(path_group)

                #U-shape
                x = random.randint(start_x,end_x)
                y = random.randint(start_y,end_y)
                v = random.randint(start_y,end_y)
                h = random.randint(start_x,end_x)

                path = pydiffvg.from_svg_path(f'M {x} {y} V {v} H {h} V {y}')
                for i in path:
                    shapes.append(i)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            stroke_color = torch.tensor([0.0,1.0,1.0,1.0]), #aqua
                                            fill_color = None)
                shape_groups.append(path_group)
                start_x += 32
                end_x += 32
            start_y += 32
            end_y += 32

    print(len(shapes),len(shape_groups))
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/painterly_batman/init.png', gamma=gamma)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = False
            color_vars.append(group.fill_color)
    else:
        for group in shape_groups:
            group.stroke_color.requires_grad = False
            group.stroke_color.data[:3].clamp_(0., 1.0) # to force black stroke
            group.stroke_color.data[-1].clamp_(0., 0.3) # opacity
            color_vars.append(group.stroke_color)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    if len(stroke_width_vars) > 0:
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), 'results/painterly_batman/iter_{}.png'.format(t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        if len(stroke_width_vars) > 0:
            width_optim.step()
        color_optim.step()
        if len(stroke_width_vars) > 0:
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
        if args.use_blob:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        else:
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/painterly_batman/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)
    
    # Render the final result.
    img = render(target.shape[1], # width
                 target.shape[0], # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/painterly_batman/final.png'.format(t), gamma=gamma)
    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/painterly_batman/iter_%d.png", "-vb", "20M",
        "results/painterly_batman/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    args = parser.parse_args()
    main(args)