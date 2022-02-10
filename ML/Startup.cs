using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.ML;
using Microsoft.OpenApi.Models;
using ML.Common;
using ML.Helpers;
using ML.Models;
using ML.Services;
using System.IO;

namespace ML
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {

            services.AddControllers();
            services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new OpenApiInfo { Title = "ML", Version = "v1" });
            });

            // Random seed for repeatable results over multiple trainings
            services.AddScoped( provider => new MLContext(seed: 2020));
            services.AddScoped<ISrCnnTrainer,SrCnnTrainer>();
            services.AddScoped<ISrCnnService,SrCnnService>();

            services.AddScoped<IFastForestModelBuilder, FastForestModelBuilder>();
            services.AddScoped<IFastForestPrediction, FastForestPrediction>();
            services.AddScoped<IFastForestService, FastForestService>();
            services.AddScoped<IMLDataLoader, MLDataLoader>();
         
            services.AddScoped<ITrainingDataService,TrainingDataService>();

            services.AddSingleton(provider =>
            {
                var env = provider.GetRequiredService<IWebHostEnvironment>();

                IDataPathRegister dataPathRegister = new DataPathRegister(
                                                        Path.Combine(env.ContentRootPath, "TrainingData"),
                                                        Path.Combine(env.ContentRootPath, "Models"));
                return dataPathRegister;
            });       
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                app.UseSwagger();
                app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "ML v1"));
            }

            app.UseHttpsRedirection();

            app.UseRouting();

            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }
}
