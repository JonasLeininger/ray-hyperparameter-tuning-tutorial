import numpy as np

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import (ColorBar,
                          FixedTicker, 
                          LinearColorMapper, 
                          PrintfTickFormatter)
from bokeh.layouts import gridplot

from trainable_simple import SimpleTrainer

def main():
    config1 = {
        "trainer_id": 0,
        "theta": np.array([0.9, 0.9]),
        "h": np.array([1., 0.]),
        "alpha": 0.01,
        "exploit": True,
        "explore": True
    }
    config2 = {
        "trainer_id": 1,
        "theta": np.array([0.9, 0.9]),
        "h": np.array([0., 1.]),
        "alpha": 0.01,
        "exploit": True,
        "explore": True
    }
    l_config = [config1, config2]

    trainers, l_scores_pbt, l_parameters_pbt = run(200, config_list=l_config)
    trainers, l_scores_exploit, l_parameters_exploit = run(200, config_list=l_config, explore=False)
    trainers, l_scores_explore, l_parameters_explore = run(200, config_list=l_config, exploit=False)
    trainers, l_scores_grid, l_parameters_grid = run(200, config_list=l_config, exploit=False, explore=False)
    
    p_pbt = plot_heatmap(l_parameters_pbt)
    p_exploit = plot_heatmap(l_parameters_exploit)
    p_explore = plot_heatmap(l_parameters_explore)
    p_grid = plot_heatmap(l_parameters_grid)

    p_score_pbt = plot_score(l_scores_pbt)
    p_score_exploit = plot_score(l_scores_exploit)
    p_score_explore = plot_score(l_scores_explore)
    p_score_grid = plot_score(l_scores_grid)

    grid = gridplot([[p_pbt, p_exploit, p_explore, p_grid], [p_score_pbt, p_score_exploit, p_score_explore, p_score_grid]])
    show(grid)


def run(steps, config_list, explore=True, exploit=True):
    l_scores = []
    l_parameters = []

    trainers = [
        SimpleTrainer(config=config_list[0]),
        SimpleTrainer(config=config_list[1])
        ]

    for step in range(steps):
        arr_score = np.zeros((2, 1))
        arr_thetas = np.zeros((2, 2))
        for trainer in trainers:
            result = trainer.step()
            arr_score[result["id"]] = result["score"]
            arr_thetas[result["id"]] = result["theta"]
        
        l_scores.append(arr_score)
        l_parameters.append(arr_thetas)

        best_trainer_id = np.argmax(arr_score)
        best_params = np.copy(arr_thetas[best_trainer_id])

        if step % 5 == 0 and step > 0:
            for trainer in trainers:
                if explore and exploit:
                    bool_explore = trainer.exploit(best_trainer_id, best_params)
                    if bool_explore:
                        trainer.explore()
                elif explore and not exploit:
                    trainer.explore()
                elif not explore and exploit:
                    trainer.exploit(best_trainer_id, best_params)
                else:
                    pass
    
    return trainers, l_scores, l_parameters

def plot_heatmap(params):
    params = np.array(params)
    trainer1 = params[...,0, :]
    trainer2 = params[...,1, :]

    N = 500
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    d = 1.2 - xx**2 - yy**2

    mapper = LinearColorMapper(palette='Viridis256', low=-0.5, high=1.2)

    p = figure(x_range=(0, 1), y_range=(0, 1),
            tooltips=[("theta_x", "$x"), ("theta_y", "$y"), ("value", "@image")])
    p.xaxis.axis_label = 'theta_x'
    p.yaxis.axis_label = 'theta_y'
    # must give a vector of image data for image parameter
    p.image(image=[d], x=0, y=0, dw=1., dh=1., 
            palette='Viridis256'
        )

    levels = np.linspace(-0.5, 1.2, 12)
    color_bar = ColorBar(color_mapper=mapper, 
                        major_label_text_font_size="8pt",
                        ticker=FixedTicker(ticks=levels),
                        formatter=PrintfTickFormatter(format='%.2f'),
                        label_standoff=6, 
                        border_line_color=None, 
                        location=(0, 0))

    p.circle(trainer1[..., 0],trainer1[..., 1], size=2, color="black", alpha=0.5)
    p.circle(trainer2[..., 0],trainer2[..., 1], size=2, color="red", alpha=0.5)
    p.add_layout(color_bar, 'right')

    # show(p)
    return p

def plot_score(scores):
    scores = np.array(scores)
    scores1 = scores[...,0, :]
    scores2 = scores[...,1, :]
    N = 200
    steps = np.linspace(0, 200, N)
    p = figure(x_range=(0, 200), y_range=(0, 1.2))
    p.xaxis.axis_label = 'steps'
    p.yaxis.axis_label = 'score'
    p.circle(steps, scores1[..., 0], size=2, color="black", alpha=0.5)
    p.circle(steps, scores2[..., 0], size=2, color="red", alpha=0.5)

    return p

if __name__ == "__main__":
    main()