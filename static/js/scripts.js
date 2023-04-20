
window.addEventListener('DOMContentLoaded', event => {

    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Uncomment Below to persist sidebar toggle between refreshes
        // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
        //     document.body.classList.toggle('sb-sidenav-toggled');
        // }
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }

});

function getWeather() {
    var url = 'https://api.openweathermap.org/data/2.5/weather?lat=30.9686&lon=76.4733&appid=23f7107e293e5bbf67c1dbd0e1e1ea70&units=metric';
    $.ajax({
      dataType: "jsonp",
      url: url,
      jsonCallback: 'jsonp',
      cache: false,
      success: function (data) {
        $('#degreeFetch').text(Math.floor(data.main.temp)+"°C");
      }
    });
  }

  setTimeout(getWeather(), 10000);
