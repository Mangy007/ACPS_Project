var btnUpload = $("#upload_file"),
		btnOuter = $(".button_outer");
	btnUpload.on("change", function(e){
		var ext = btnUpload.val().split('.').pop().toLowerCase();
		if($.inArray(ext, ['mp3', 'wav']) == -1) {
			$(".error_msg").text("Not an Audio Format...");
		} else {
			$(".error_msg").text("");
			btnOuter.addClass("file_uploading");
			setTimeout(function(){
				btnOuter.addClass("file_uploaded");
			},3000);
			var uploadedFile = URL.createObjectURL(e.target.files[0]);
			// console.log("aaya: ", uploadedFile);
			var audioFile = new File([ e.target.files[0] ], btnUpload.val());      
			var form    = new FormData();
			form.append("audio", audioFile);
			
    	$.ajax({
	        type: 'POST',
	        url:"/upload-audio",
	        data: form,
			dataType: "json",
	        processData: false, 
	        contentType: false, 
	        success: function(returnval) {
	        	console.log("agaya");
	        	console.log(returnval);
				$("#uploaded_view").append('<audio  controls autoplay muted> <source src="'+uploadedFile+'" type="audio/mpeg"></audio>').addClass("show");
				if(returnval['status']) {
					// if result is true print the button for classification
					if(returnval['result']) {
						// returns distress
						$("#uploaded_view").append('<button class="btn danger">Distress</button>').addClass("show");
					}
					else {
						// no distress detected
						$("#uploaded_view").append('<button class="btn success">No Distress</button>').addClass("show");
					}
				}
				else {
					$("#uploaded_view").append('<button class="btn warning">500 error</button>').addClass("show");
				}
				
	             // $("#show1").html(returnval);
	             // $('#show1').show();
	         }
	    });

			
		}
	});
	$(".file_remove").on("click", function(e){
		$("#uploaded_view").removeClass("show");
		$("#uploaded_view").find("audio").remove();
		$("#uploaded_view").find("button").remove();
		btnOuter.removeClass("file_uploading");
		btnOuter.removeClass("file_uploaded");
	});