use eframe::egui;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

// Add a new struct to hold processing metrics
#[derive(Clone, Debug)]
struct ProcessingMetrics {
    frame_path: String,
    change_count: i32,
    current_latency: i64,
    average_latency: i64,
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("SAR Change Detection - CUDA Accelerated")
            .with_min_inner_size([800.0, 600.0])
            .with_resizable(true),
        ..Default::default()
    };

    eframe::run_native(
        "SAR Change Detection",
        options,
        Box::new(|cc| {
            // Setup custom style for modern UI
            setup_custom_style(&cc.egui_ctx);
            let mut app = PipelineApp::default();
            // Scan for existing output frames on startup
            app.scan_existing_output_frames();
            // Set startup success message
            app.startup_message = "✅ Application started successfully! Ready to process videos.".to_string();
            Ok(Box::new(app))
        }),
    )
}

fn setup_custom_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    
    // Modern dark theme colors
    style.visuals.dark_mode = true;
    style.visuals.window_fill = egui::Color32::from_rgb(18, 18, 20);
    style.visuals.panel_fill = egui::Color32::from_rgb(25, 25, 28);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(35, 35, 40);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(15, 15, 17);
    
    // Accent colors (modern blue)
    style.visuals.selection.bg_fill = egui::Color32::from_rgb(70, 130, 255);
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(40, 40, 45);
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 60, 70);
    
    // Button styling
    style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(35, 35, 40);
    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(45, 45, 50);
    
    // Enhanced spacing and sizing
    style.spacing.item_spacing = egui::vec2(12.0, 8.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.window_margin = egui::Margin::same(20.0);
    style.spacing.menu_margin = egui::Margin::same(8.0);
    
    // Rounded corners for modern look
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(8.0);
    style.visuals.window_rounding = egui::Rounding::same(12.0);
    
    // Better text rendering (egui 0.28.1 uses methods instead of fields)
    style.visuals.override_text_color = Some(egui::Color32::from_rgb(220, 220, 220));
    
    ctx.set_style(style);
}

// Add metrics fields to PipelineApp
struct PipelineApp {
    selected_video_path: Option<PathBuf>,
    output_folder_path: Option<PathBuf>,
    processing_status: ProcessingStatus,
    status_receiver: Option<mpsc::Receiver<String>>,
    current_status: String,
    processed_videos: Vec<ProcessedVideo>,
    output_frames: Vec<PathBuf>,
    selected_frame: Option<PathBuf>,
    startup_message: String,
    cancel_sender: Option<mpsc::Sender<bool>>,
    frame_receiver: Option<mpsc::Receiver<Vec<PathBuf>>>,
    compilation_cancel_sender: Option<mpsc::Sender<bool>>,
    is_compiling: bool,
    frame_rate: f32,
    frame_rate_options: Vec<(String, f32)>,
    selected_frame_rate_index: usize,
    // Slideshow fields
    slideshow_active: bool,
    slideshow_current_index: usize,
    slideshow_last_change: std::time::Instant,
    slideshow_interval: f32,
    // Real-time frame monitoring
    auto_slideshow_on_generation: bool,
    last_frame_count: usize,
    frame_monitor_last_check: std::time::Instant,
    // Image preview
    loaded_images: HashMap<PathBuf, egui::TextureHandle>,
    preview_size: egui::Vec2,
    show_preview: bool,
    // Add metrics fields
    metrics_receiver: Option<mpsc::Receiver<ProcessingMetrics>>,
    latest_metrics: Option<ProcessingMetrics>,
    all_metrics: Vec<ProcessingMetrics>,
    total_frames_processed: usize,
    overall_average_latency: f64,
}

impl Default for PipelineApp {
    fn default() -> Self {
        let frame_rate_options = vec![
            ("0.1 FPS (Very Slow)".to_string(), 0.1),
            ("0.25 FPS (Ultra Slow)".to_string(), 0.25),
            ("0.5 FPS (Slow)".to_string(), 0.5),
            ("1 FPS (Default)".to_string(), 1.0),
            ("2 FPS (Fast)".to_string(), 2.0),
            ("4 FPS (Very Fast)".to_string(), 4.0),
            ("5 FPS (High)".to_string(), 5.0),
            ("10 FPS (Very High)".to_string(), 10.0),
            ("15 FPS (Maximum)".to_string(), 15.0),
        ];
        
        Self {
            selected_video_path: None,
            output_folder_path: None,
            processing_status: ProcessingStatus::default(),
            status_receiver: None,
            current_status: String::new(),
            processed_videos: Vec::new(),
            output_frames: Vec::new(),
            selected_frame: None,
            startup_message: String::new(),
            cancel_sender: None,
            frame_receiver: None,
            compilation_cancel_sender: None,
            is_compiling: false,
            frame_rate: 1.0, // Default to 1 FPS
            frame_rate_options,
            selected_frame_rate_index: 3, // Default to "1 FPS (Default)"
            // Slideshow defaults
            slideshow_active: false,
            slideshow_current_index: 0,
            slideshow_last_change: std::time::Instant::now(),
            slideshow_interval: 2.0, // 2 seconds between slides
            // Real-time monitoring defaults
            auto_slideshow_on_generation: true, // Auto-start slideshow when generating
            last_frame_count: 0,
            frame_monitor_last_check: std::time::Instant::now(),
            // Image preview defaults
            loaded_images: HashMap::new(),
            preview_size: egui::Vec2::new(800.0, 1000.0),
            show_preview: true,
            // Initialize metrics fields
            metrics_receiver: None,
            latest_metrics: None,
            all_metrics: Vec::new(),
            total_frames_processed: 0,
            overall_average_latency: 0.0,
        }
    }
}

#[derive(Clone)]
struct ProcessedVideo {
    name: String,
    path: PathBuf,
    status: String,
    timestamp: SystemTime,
}

impl ProcessedVideo {
    fn format_timestamp(&self) -> String {
        use std::time::UNIX_EPOCH;
        if let Ok(duration) = self.timestamp.duration_since(UNIX_EPOCH) {
            let secs = duration.as_secs();
            if let Some(datetime) = chrono::DateTime::from_timestamp(secs as i64, 0) {
                return datetime.format("%Y-%m-%d %H:%M:%S").to_string();
            }
        }
        "Unknown".to_string()
    }
}

#[derive(Default, PartialEq)]
enum ProcessingStatus {
    #[default]
    Idle,
    Processing,
    Completed,
    Error(String),
    Cancelled,
}

impl eframe::App for PipelineApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for metrics updates
        if let Some(metrics_receiver) = &self.metrics_receiver {
            while let Ok(metrics) = metrics_receiver.try_recv() {
                self.latest_metrics = Some(metrics.clone());
                self.all_metrics.push(metrics);
                self.total_frames_processed = self.all_metrics.len();
                
                // Calculate overall average
                if !self.all_metrics.is_empty() {
                    self.overall_average_latency = self.all_metrics.iter()
                        .map(|m| m.current_latency as f64)
                        .sum::<f64>() / self.all_metrics.len() as f64;
                }
                
                ctx.request_repaint();
            }
        }

        // Real-time frame monitoring during processing
        if matches!(self.processing_status, ProcessingStatus::Processing) || self.auto_slideshow_on_generation {
            // Check for new frames every 500ms
            if self.frame_monitor_last_check.elapsed().as_millis() > 500 {
                self.check_for_new_frames();
                self.frame_monitor_last_check = std::time::Instant::now();
            }
        }
        
        // Handle automatic slideshow timing (only for manual slideshow, not during preview)
        if self.slideshow_active && !self.output_frames.is_empty() && !matches!(self.processing_status, ProcessingStatus::Processing) {
            let elapsed = self.slideshow_last_change.elapsed().as_secs_f32();
            if elapsed >= self.slideshow_interval {
                self.advance_slideshow();
                self.slideshow_last_change = std::time::Instant::now();
            }
            // Request continuous updates for smooth slideshow
            ctx.request_repaint();
        }
        
        // Check for status updates
        if let Some(receiver) = &self.status_receiver {
            if let Ok(status) = receiver.try_recv() {
                self.current_status = status.clone();
                if status.contains("successfully") {
                    self.processing_status = ProcessingStatus::Completed;
                    
                    // Check if this was a compilation completion
                    if self.is_compiling {
                        self.is_compiling = false;
                        self.compilation_cancel_sender = None;
                    } else {
                        // This was video processing completion
                        if let Some(last_video) = self.processed_videos.last_mut() {
                            last_video.status = "Completed".to_string();
                        }
                        // Scan for output frames
                        self.scan_output_frames();
                        // Clear cancel sender when completed
                        self.cancel_sender = None;
                        // Clear frame receiver when completed
                        self.frame_receiver = None;
                    }
                } else if status.contains("Error") || status.contains("Cancelled") {
                    let is_cancelled = status.contains("Cancelled");
                    if is_cancelled {
                        self.processing_status = ProcessingStatus::Cancelled;
                    } else {
                        self.processing_status = ProcessingStatus::Error(status.clone());
                    }
                    
                    // Check if this was a compilation error/cancellation
                    if self.is_compiling {
                        self.is_compiling = false;
                        self.compilation_cancel_sender = None;
                    } else {
                        // This was video processing error/cancellation
                        if let Some(last_video) = self.processed_videos.last_mut() {
                            last_video.status = if is_cancelled { "Cancelled".to_string() } else { "Failed".to_string() };
                        }
                        // Clear cancel sender when error/cancelled
                        self.cancel_sender = None;
                        // Clear frame receiver when error/cancelled
                        self.frame_receiver = None;
                    }
                }
            }
        }

        // Check for frame updates
        if let Some(frame_receiver) = &self.frame_receiver {
            if let Ok(new_frames) = frame_receiver.try_recv() {
                self.output_frames = new_frames;
                ctx.request_repaint(); // Force UI update
            }
        }

        // Top panel for header
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    // App icon and title
                    ui.add(egui::Label::new(
                        egui::RichText::new("🎬")
                            .size(32.0)
                    ));
                    ui.add_space(12.0);
                    ui.vertical(|ui| {
                        ui.add(egui::Label::new(
                            egui::RichText::new("SAR Video Processing Pipeline")
                                .size(24.0)
                                .strong()
                        ));
                        ui.add(egui::Label::new(
                            egui::RichText::new("CUDA-Accelerated SAR Video Processing")
                                .size(14.0)
                                .color(egui::Color32::from_rgb(160, 160, 160))
                        ));
                    });
                });
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(20.0);
                    // Status indicator
                    match &self.processing_status {
                        ProcessingStatus::Idle => {
                            ui.add(egui::Label::new(
                                egui::RichText::new("🟢 Running Successfully")
                                    .color(egui::Color32::GREEN)
                            ));
                        }
                        ProcessingStatus::Processing => {
                            ui.add(egui::Label::new(
                                egui::RichText::new("🟡 Processing")
                                    .color(egui::Color32::YELLOW)
                            ));
                        }
                        ProcessingStatus::Completed => {
                            ui.add(egui::Label::new(
                                egui::RichText::new("🟢 Completed")
                                    .color(egui::Color32::GREEN)
                            ));
                        }
                        ProcessingStatus::Error(_) => {
                            ui.add(egui::Label::new(
                                egui::RichText::new("🔴 Error")
                                    .color(egui::Color32::RED)
                            ));
                        }
                        ProcessingStatus::Cancelled => {
                            ui.add(egui::Label::new(
                                egui::RichText::new("⏹ Cancelled")
                                    .color(egui::Color32::from_rgb(255, 165, 0))
                            ));
                        }
                    }
                });
            });
            ui.add_space(10.0);
        });

        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(20.0);
            
            // Two-column layout for better organization
            ui.horizontal_top(|ui| {
                // Left column - Input controls
                ui.vertical(|ui| {
                    ui.set_min_width(400.0);
                    ui.set_max_width(450.0);
                    
                    // Wrap left column content in a scroll area
                    egui::ScrollArea::vertical()
                        .id_source("left_column_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                    
                    // Video selection card
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(30, 30, 35))
                        .rounding(12.0)
                        .inner_margin(egui::Margin::same(20.0))
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new("📹 Video Input")
                                    .size(18.0)
                                    .strong()
                            ));
                            ui.add_space(12.0);
                            
                            // Large, prominent video selection button
                            let video_button = egui::Button::new(
                                egui::RichText::new("📁 Select Video File")
                                    .size(16.0)
                            )
                            .min_size(egui::vec2(350.0, 50.0));
                            
                            if ui.add(video_button).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Video Files", &["mp4", "avi", "mov", "mkv", "webm"])
                                    .set_title("Select Video File")
                                    .pick_file()
                                {
                                    self.selected_video_path = Some(path);
                                }
                            }
                            
                            ui.add_space(8.0);
                            
                            // Display selected file with better styling
                            if let Some(path) = &self.selected_video_path {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(40, 40, 45))
                                    .rounding(8.0)
                                    .inner_margin(egui::Margin::same(12.0))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.add(egui::Label::new("✅"));
                                            ui.add(egui::Label::new(
                                                egui::RichText::new(
                                                    path.file_name().unwrap_or_default().to_string_lossy()
                                                )
                                                .color(egui::Color32::from_rgb(120, 200, 120))
                                            ));
                                        });
                                    });
                            } else {
                                ui.add(egui::Label::new(
                                    egui::RichText::new("No video file selected")
                                        .color(egui::Color32::from_rgb(140, 140, 140))
                                        .italics()
                                ));
                            }
                        });

                    ui.add_space(20.0);

                    // Output folder selection card
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(30, 30, 35))
                        .rounding(12.0)
                        .inner_margin(egui::Margin::same(20.0))
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new("📂 Output Destination")
                                    .size(18.0)
                                    .strong()
                            ));
                            ui.add_space(12.0);
                            
                            let folder_button = egui::Button::new(
                                egui::RichText::new("📁 Select Output Folder")
                                    .size(16.0)
                            )
                            .min_size(egui::vec2(350.0, 50.0));
                            
                            if ui.add(folder_button).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Select Output Folder")
                                    .pick_folder() {
                                    self.output_folder_path = Some(path);
                                }
                            }
                            
                            ui.add_space(8.0);
                            
                            if let Some(path) = &self.output_folder_path {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(40, 40, 45))
                                    .rounding(8.0)
                                    .inner_margin(egui::Margin::same(12.0))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.add(egui::Label::new("✅"));
                                            ui.add(egui::Label::new(
                                                egui::RichText::new(
                                                    path.file_name().unwrap_or_default().to_string_lossy()
                                                )
                                                .color(egui::Color32::from_rgb(120, 200, 120))
                                            ));
                                        });
                                    });
                            } else {
                                ui.add(egui::Label::new(
                                    egui::RichText::new("No output folder selected")
                                        .color(egui::Color32::from_rgb(140, 140, 140))
                                        .italics()
                                ));
                            }
                        });

                    ui.add_space(20.0);

                    // Processing controls card
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(30, 30, 35))
                        .rounding(12.0)
                        .inner_margin(egui::Margin::same(20.0))
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new("🚀 Processing Controls")
                                    .size(18.0)
                                    .strong()
                            ));
                            ui.add_space(12.0);
                            
                            // Frame rate selection dropdown
                            ui.horizontal(|ui| {
                                ui.add(egui::Label::new(
                                    egui::RichText::new("⚡ Frame Rate (FPS):")
                                        .size(14.0)
                                        .strong()
                                ));
                                ui.add_space(8.0);
                                
                                // Frame rate dropdown
                                let selected_text = &self.frame_rate_options[self.selected_frame_rate_index].0;
                                egui::ComboBox::from_label("")
                                    .selected_text(selected_text)
                                    .width(200.0)
                                    .show_ui(ui, |ui| {
                                        for (i, (display_text, _value)) in self.frame_rate_options.iter().enumerate() {
                                            let response = ui.selectable_value(&mut self.selected_frame_rate_index, i, display_text);
                                            if response.clicked() {
                                                // Update frame_rate when selection changes
                                                self.frame_rate = self.frame_rate_options[i].1;
                                            }
                                        }
                                    });
                            });
                            
                            ui.add_space(4.0);
                            ui.add(egui::Label::new(
                                egui::RichText::new("📝 Controls how many frames per second are extracted from the video")
                                    .size(12.0)
                                    .color(egui::Color32::from_rgb(160, 160, 160))
                                    .italics()
                            ));
                            
                            ui.add_space(12.0);
                            
                            let can_process = self.selected_video_path.is_some() 
                                && self.output_folder_path.is_some() 
                                && self.processing_status != ProcessingStatus::Processing;

                            // Main process button
                            let process_button = if can_process {
                                egui::Button::new(
                                    egui::RichText::new("▶ Start Processing")
                                        .size(18.0)
                                        .color(egui::Color32::WHITE)
                                )
                                .fill(egui::Color32::from_rgb(70, 130, 255))
                                .min_size(egui::vec2(350.0, 60.0))
                            } else {
                                egui::Button::new(
                                    egui::RichText::new("▶ Start Processing")
                                        .size(18.0)
                                )
                                .min_size(egui::vec2(350.0, 60.0))
                            };

                            if ui.add_enabled(can_process, process_button).clicked() {
                                self.start_processing();
                            }

                            // Cancel button (only show when processing)
                            if self.processing_status == ProcessingStatus::Processing {
                                ui.add_space(8.0);
                                if ui.add(
                                    egui::Button::new(
                                        egui::RichText::new("⏹ Cancel Processing")
                                            .size(16.0)
                                            .color(egui::Color32::WHITE)
                                    )
                                    .fill(egui::Color32::from_rgb(200, 50, 50))
                                    .min_size(egui::vec2(350.0, 45.0))
                                ).clicked() {
                                    self.cancel_processing();
                                }
                            }

                            ui.add_space(12.0);
                            
                            // Secondary action buttons
                            ui.horizontal(|ui| {
                                if ui.add(
                                    egui::Button::new(
                                        egui::RichText::new("🗑 Clear")
                                            .size(14.0)
                                    )
                                    .min_size(egui::vec2(100.0, 35.0))
                                ).clicked() {
                                    self.selected_video_path = None;
                                    self.output_folder_path = None;
                                    self.processing_status = ProcessingStatus::Idle;
                                    self.current_status.clear();
                                    self.output_frames.clear();
                                    self.selected_frame = None;
                                    self.cancel_sender = None;
                                }

                                ui.add_space(8.0);

                                if ui.add(
                                    egui::Button::new(
                                        egui::RichText::new("📋 Clear History")
                                            .size(14.0)
                                    )
                                    .min_size(egui::vec2(120.0, 35.0))
                                ).clicked() {
                                    self.processed_videos.clear();
                                    self.output_frames.clear();
                                    self.selected_frame = None;
                                }
                            });

                            ui.add_space(8.0);

                            // CUDA compilation button
                            ui.horizontal(|ui| {
                                if ui.add(
                                    egui::Button::new(
                                        egui::RichText::new("🔧 Compile CUDA Code")
                                            .size(14.0)
                                    )
                                    .fill(egui::Color32::from_rgb(45, 65, 85))
                                    .min_size(egui::vec2(350.0, 40.0))
                                ).clicked() {
                                    self.compile_cuda_code();
                                }
                            });

                            // Cancel compilation button (only show when compiling)
                            if self.is_compiling && self.processing_status == ProcessingStatus::Processing {
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    if ui.add(
                                        egui::Button::new(
                                            egui::RichText::new("⏹ Cancel Compilation")
                                                .size(14.0)
                                                .color(egui::Color32::WHITE)
                                        )
                                        .fill(egui::Color32::from_rgb(200, 50, 50))
                                        .min_size(egui::vec2(350.0, 35.0))
                                    ).clicked() {
                                        self.cancel_compilation();
                                    }
                                });
                            }
                        });
                    }); // Close scroll area
                });

                ui.add_space(20.0);

                // Right column - Status and results
                ui.vertical(|ui| {
                    ui.set_min_width(400.0);
                    
                    // Wrap right column content in a scroll area
                    egui::ScrollArea::vertical()
                        .id_source("right_column_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                    
                    // Performance Metrics Card (NEW)
                    if self.latest_metrics.is_some() || !self.all_metrics.is_empty() {
                        egui::Frame::none()
                            .fill(egui::Color32::from_rgb(30, 30, 35))
                            .rounding(12.0)
                            .inner_margin(egui::Margin::same(20.0))
                            .show(ui, |ui| {
                                ui.add(egui::Label::new(
                                    egui::RichText::new("⚡ Performance Metrics")
                                        .size(18.0)
                                        .strong()
                                ));
                                ui.add_space(12.0);
                                
                                // Real-time metrics display
                                if let Some(metrics) = &self.latest_metrics {
                                    egui::Grid::new("metrics_grid")
                                        .num_columns(2)
                                        .spacing([20.0, 8.0])
                                        .show(ui, |ui| {
                                            // Current frame metrics
                                            ui.add(egui::Label::new(
                                                egui::RichText::new("🎯 Current Frame:")
                                                    .strong()
                                            ));
                                            ui.add(egui::Label::new(
                                                egui::RichText::new(format!("{} ms", metrics.current_latency))
                                                    .color(egui::Color32::from_rgb(120, 200, 255))
                                            ));
                                            ui.end_row();
                                            
                                            // Running average
                                            ui.add(egui::Label::new(
                                                egui::RichText::new("📊 Running Average:")
                                                    .strong()
                                            ));
                                            ui.add(egui::Label::new(
                                                egui::RichText::new(format!("{} ms", metrics.average_latency))
                                                    .color(egui::Color32::from_rgb(100, 255, 100))
                                                    .strong()
                                            ));
                                            ui.end_row();
                                            
                                            // Changes detected
                                            // ui.add(egui::Label::new(
                                            //     egui::RichText::new("🔍 Changes Found:")
                                            //         .strong()
                                            // ));
                                            // ui.add(egui::Label::new(
                                            //     egui::RichText::new(format!("{}", metrics.change_count))
                                            //         .color(egui::Color32::from_rgb(255, 200, 100))
                                            // ));
                                            // ui.end_row();
                                            
                                            // Total frames processed
                                            ui.add(egui::Label::new(
                                                egui::RichText::new("📹 Frames Processed:")
                                                    .strong()
                                            ));
                                            ui.add(egui::Label::new(
                                                egui::RichText::new(format!("{}", self.total_frames_processed))
                                                    .color(egui::Color32::from_rgb(200, 200, 255))
                                            ));
                                            ui.end_row();
                                            
                                            // Overall average (calculated by Rust)
                                            if self.total_frames_processed > 0 {
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("🎯 Overall Average:")
                                                        .strong()
                                                ));
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new(format!("{:.1} ms", self.overall_average_latency))
                                                        .color(egui::Color32::from_rgb(255, 100, 255))
                                                        .strong()
                                                ));
                                                ui.end_row();
                                            }
                                        });
                                    
                                    ui.add_space(8.0);
                                    ui.separator();
                                    ui.add_space(8.0);
                                    
                                    // Performance summary
                                    let fps_estimate = if metrics.average_latency > 0 {
                                        1000.0 / metrics.average_latency as f64
                                    } else {
                                        0.0
                                    };
                                    
                                    ui.horizontal(|ui| {
                                        ui.add(egui::Label::new(
                                            egui::RichText::new("🚀 Throughput:")
                                                .strong()
                                        ));
                                        ui.add(egui::Label::new(
                                            egui::RichText::new(format!("{:.2} FPS", fps_estimate))
                                                .color(egui::Color32::from_rgb(255, 255, 100))
                                        ));
                                    });
                                    
                                    // Performance status indicator
                                    ui.add_space(4.0);
                                    let (status_text, status_color) = if metrics.average_latency < 100 {
                                        ("🟢 Excellent Performance", egui::Color32::GREEN)
                                    } else if metrics.average_latency < 500 {
                                        ("🟡 Good Performance", egui::Color32::YELLOW)
                                    } else if metrics.average_latency < 1000 {
                                        ("🟠 Fair Performance", egui::Color32::from_rgb(255, 165, 0))
                                    } else {
                                        ("🔴 Slow Performance", egui::Color32::RED)
                                    };
                                    
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(status_text)
                                            .color(status_color)
                                            .italics()
                                    ));
                                } else {
                                    ui.add(egui::Label::new(
                                        egui::RichText::new("No metrics available yet...")
                                            .color(egui::Color32::from_rgb(140, 140, 140))
                                            .italics()
                                    ));
                                }
                            });
                        
                        ui.add_space(20.0);
                    }
                    
                    // Status display card
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(30, 30, 35))
                        .rounding(12.0)
                        .inner_margin(egui::Margin::same(20.0))
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(
                                egui::RichText::new("📊 Processing Status")
                                    .size(18.0)
                                    .strong()
                            ));
                            ui.add_space(12.0);
                            
                            match &self.processing_status {
                                ProcessingStatus::Idle => {
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(20, 50, 20))
                                        .rounding(8.0)
                                        .inner_margin(egui::Margin::same(16.0))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("✅")
                                                        .size(20.0)
                                                ));
                                                ui.vertical(|ui| {
                                                    ui.add(egui::Label::new(
                                                        egui::RichText::new("Application Running Successfully")
                                                            .size(16.0)
                                                            .color(egui::Color32::GREEN)
                                                    ));
                                                    ui.add(egui::Label::new(
                                                        egui::RichText::new("Ready to process video files")
                                                            .size(14.0)
                                                            .color(egui::Color32::from_rgb(180, 255, 180))
                                                    ));
                                                    if !self.output_frames.is_empty() {
                                                        ui.add(egui::Label::new(
                                                            egui::RichText::new(format!("Found {} existing output frames", self.output_frames.len()))
                                                                .size(12.0)
                                                                .color(egui::Color32::from_rgb(160, 200, 160))
                                                        ));
                                                    }
                                                });
                                            });
                                        });
                                }
                                ProcessingStatus::Processing => {
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(50, 40, 20))
                                        .rounding(8.0)
                                        .inner_margin(egui::Margin::same(16.0))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.spinner();
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("Processing video...")
                                                        .size(16.0)
                                                        .color(egui::Color32::YELLOW)
                                                ));
                                            });
                                            if !self.current_status.is_empty() {
                                                ui.add_space(8.0);
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new(&self.current_status)
                                                        .size(14.0)
                                                        .color(egui::Color32::from_rgb(200, 200, 160))
                                                ));
                                            }
                                        });
                                }
                                ProcessingStatus::Completed => {
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(20, 50, 20))
                                        .rounding(8.0)
                                        .inner_margin(egui::Margin::same(16.0))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("✅")
                                                        .size(20.0)
                                                ));
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("Processing completed successfully!")
                                                        .size(16.0)
                                                        .color(egui::Color32::GREEN)
                                                ));
                                            });
                                        });
                                }
                                ProcessingStatus::Error(error) => {
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(50, 20, 20))
                                        .rounding(8.0)
                                        .inner_margin(egui::Margin::same(16.0))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("❌")
                                                        .size(20.0)
                                                ));
                                                ui.vertical(|ui| {
                                                    ui.add(egui::Label::new(
                                                        egui::RichText::new("Processing Error")
                                                            .size(16.0)
                                                            .color(egui::Color32::RED)
                                                    ));
                                                    ui.add(egui::Label::new(
                                                        egui::RichText::new(error)
                                                            .size(14.0)
                                                            .color(egui::Color32::from_rgb(255, 180, 180))
                                                    ));
                                                });
                                            });
                                        });
                                }
                                ProcessingStatus::Cancelled => {
                                    egui::Frame::none()
                                        .fill(egui::Color32::from_rgb(50, 35, 20))
                                        .rounding(8.0)
                                        .inner_margin(egui::Margin::same(16.0))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("⏹")
                                                        .size(20.0)
                                                ));
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new("Processing Cancelled")
                                                        .size(16.0)
                                                        .color(egui::Color32::from_rgb(255, 165, 0))
                                                ));
                                            });
                                            ui.add_space(4.0);
                                            ui.add(egui::Label::new(
                                                egui::RichText::new("Processing was stopped by user request")
                                                    .size(14.0)
                                                    .color(egui::Color32::from_rgb(255, 200, 150))
                                            ));
                                        });
                                }
                            }
                        });

                    ui.add_space(20.0);

                    // Processing history card
                    if !self.processed_videos.is_empty() {
                        egui::Frame::none()
                            .fill(egui::Color32::from_rgb(30, 30, 35))
                            .rounding(12.0)
                            .inner_margin(egui::Margin::same(20.0))
                            .show(ui, |ui| {
                                ui.add(egui::Label::new(
                                    egui::RichText::new("📁 Processing History")
                                        .size(18.0)
                                        .strong()
                                ));
                                ui.add_space(12.0);
                                
                                egui::ScrollArea::vertical()
                                    .id_source("processing_history_scroll")
                                    .max_height(300.0)
                                    .show(ui, |ui| {
                                        for video in &self.processed_videos {
                                            egui::Frame::none()
                                                .fill(egui::Color32::from_rgb(40, 40, 45))
                                                .rounding(8.0)
                                                .inner_margin(egui::Margin::same(12.0))
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        // Status icon with color
                                                        let (icon, color) = match video.status.as_str() {
                                                            "Completed" => ("✅", egui::Color32::GREEN),
                                                            "Processing..." => ("🔄", egui::Color32::YELLOW),
                                                            "Failed" => ("❌", egui::Color32::RED),
                                                            "Cancelled" => ("⏹", egui::Color32::from_rgb(255, 165, 0)), // Orange
                                                            _ => ("❓", egui::Color32::GRAY),
                                                        };
                                                        ui.add(egui::Label::new(
                                                            egui::RichText::new(icon)
                                                                .size(16.0)
                                                                .color(color)
                                                        ));
                                                        
                                                        ui.vertical(|ui| {
                                                            ui.add(egui::Label::new(
                                                                egui::RichText::new(&video.name)
                                                                    .size(14.0)
                                                                    .strong()
                                                            ));
                                                            ui.add(egui::Label::new(
                                                                egui::RichText::new(&video.status)
                                                                    .size(12.0)
                                                                    .color(egui::Color32::from_rgb(160, 160, 160))
                                                            ));
                                                        });
                                                        
                                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                            if video.status == "Completed" {
                                                                if ui.add(
                                                                    egui::Button::new(
                                                                        egui::RichText::new("📂 Open")
                                                                            .size(12.0)
                                                                    )
                                                                    .min_size(egui::vec2(60.0, 30.0))
                                                                ).clicked() {
                                                                    let _ = Command::new("xdg-open")
                                                                        .arg(video.path.parent().unwrap_or(&video.path))
                                                                        .spawn();
                                                                }
                                                            }
                                                        });
                                                    });
                                                });
                                            ui.add_space(8.0);
                                        }
                                    });
                            });
                    }

                    // Output frames preview card
                    if !self.output_frames.is_empty() {
                        ui.add_space(20.0);
                        egui::Frame::none()
                            .fill(egui::Color32::from_rgb(30, 30, 35))
                            .rounding(12.0)
                            .inner_margin(egui::Margin::same(20.0))
                            .show(ui, |ui| {
                                ui.add(egui::Label::new(
                                    egui::RichText::new("🖼️ Output Frames")
                                        .size(18.0)
                                        .strong()
                                ));
                                ui.add_space(12.0);
                                
                                ui.horizontal(|ui| {
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(format!("📊 {} frames generated", self.output_frames.len()))
                                            .size(14.0)
                                            .color(egui::Color32::from_rgb(120, 200, 120))
                                    ));
                                    
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.add(
                                            egui::Button::new(
                                                egui::RichText::new("📂 Open Folder")
                                                    .size(12.0)
                                            )
                                            .min_size(egui::vec2(100.0, 30.0))
                                        ).clicked() {
                                            if let Some(output_path) = &self.output_folder_path {
                                                let _ = Command::new("xdg-open")
                                                    .arg(output_path)
                                                    .spawn();
                                            }
                                        }
                                    });
                                });
                                
                                ui.add_space(8.0);
                                
                                // Frame list with preview
                                egui::ScrollArea::vertical()
                                    .id_source("output_frames_scroll")
                                    .max_height(200.0)
                                    .show(ui, |ui| {
                                        egui::Grid::new("frames_grid")
                                            .num_columns(3)
                                            .spacing([8.0, 4.0])
                                            .show(ui, |ui| {
                                                for (i, frame_path) in self.output_frames.iter().enumerate() {
                                                    let frame_name = frame_path.file_name()
                                                        .unwrap_or_default()
                                                        .to_string_lossy();
                                                    
                                                    let is_selected = Some(frame_path) == self.selected_frame.as_ref();
                                                    let button_color = if is_selected {
                                                        egui::Color32::from_rgb(70, 130, 255)
                                                    } else {
                                                        egui::Color32::from_rgb(50, 50, 55)
                                                    };
                                                    
                                                    if ui.add(
                                                        egui::Button::new(
                                                            egui::RichText::new(frame_name.as_ref())
                                                                .size(12.0)
                                                        )
                                                        .fill(button_color)
                                                        .min_size(egui::vec2(120.0, 25.0))
                                                    ).clicked() {
                                                        self.selected_frame = Some(frame_path.clone());
                                                        // Open image with default viewer
                                                        let _ = Command::new("xdg-open")
                                                            .arg(frame_path)
                                                            .spawn();
                                                    }
                                                    
                                                    // Show 3 items per row
                                                    if (i + 1) % 3 == 0 {
                                                        ui.end_row();
                                                    }
                                                }
                                            });
                                    });
                                
                                // Slideshow controls
                                ui.add_space(12.0);
                                ui.separator();
                                ui.add_space(8.0);
                                
                                ui.horizontal(|ui| {
                                    ui.add(egui::Label::new(
                                        egui::RichText::new("🎬 Slideshow")
                                            .size(14.0)
                                            .strong()
                                    ));
                                    
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        // Reset button
                                        if ui.add(
                                            egui::Button::new("⏮")
                                                .min_size(egui::vec2(30.0, 25.0))
                                        ).clicked() {
                                            self.reset_slideshow();
                                        }
                                        
                                        // Previous button
                                        if ui.add(
                                            egui::Button::new("⏪")
                                                .min_size(egui::vec2(30.0, 25.0))
                                        ).clicked() {
                                            self.previous_slide();
                                        }
                                        
                                        // Next button
                                        if ui.add(
                                            egui::Button::new("⏩")
                                                .min_size(egui::vec2(30.0, 25.0))
                                        ).clicked() {
                                            self.advance_slideshow();
                                        }
                                        
                                        // Play/Pause button
                                        let play_pause_text = if self.slideshow_active { "⏸" } else { "▶" };
                                        let play_pause_color = if self.slideshow_active {
                                            egui::Color32::from_rgb(255, 100, 100)
                                        } else {
                                            egui::Color32::from_rgb(100, 255, 100)
                                        };
                                        
                                        if ui.add(
                                            egui::Button::new(
                                                egui::RichText::new(play_pause_text)
                                                    .color(play_pause_color)
                                            )
                                            .min_size(egui::vec2(40.0, 25.0))
                                        ).clicked() {
                                            self.toggle_slideshow();
                                        }
                                    });
                                });
                                
                                // Slideshow speed control
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    ui.add(egui::Label::new("Speed:"));
                                    ui.add(egui::Slider::new(&mut self.slideshow_interval, 0.5..=10.0)
                                        .text("seconds")
                                        .step_by(0.5));
                                    
                                    if self.slideshow_active {
                                        ui.add(egui::Label::new(
                                            egui::RichText::new(format!("Frame {}/{}", 
                                                self.slideshow_current_index + 1,
                                                self.output_frames.len()
                                            ))
                                            .size(12.0)
                                            .color(egui::Color32::from_rgb(120, 200, 120))
                                        ));
                                    }
                                    
                                    // Show real-time generation status
                                    if matches!(self.processing_status, ProcessingStatus::Processing) {
                                        ui.add(egui::Label::new(
                                            egui::RichText::new("🔄 Generating...")
                                                .size(11.0)
                                                .color(egui::Color32::from_rgb(255, 200, 100))
                                        ));
                                    }
                                });
                                
                                // Auto-slideshow toggle
                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    let auto_text = if self.auto_slideshow_on_generation { 
                                        "🔄 Auto-slideshow: ON" 
                                    } else { 
                                        "⏹️ Auto-slideshow: OFF" 
                                    };
                                    let auto_color = if self.auto_slideshow_on_generation {
                                        egui::Color32::from_rgb(100, 255, 100)
                                    } else {
                                        egui::Color32::from_rgb(160, 160, 160)
                                    };
                                    
                                    if ui.add(
                                        egui::Button::new(
                                            egui::RichText::new(auto_text)
                                                .size(12.0)
                                                .color(auto_color)
                                        )
                                        .min_size(egui::vec2(150.0, 25.0))
                                    ).clicked() {
                                        self.toggle_auto_slideshow();
                                    }
                                    
                                    ui.add(egui::Label::new(
                                        egui::RichText::new("(Auto-start after processing completes)")
                                            .size(10.0)
                                            .color(egui::Color32::from_rgb(140, 140, 140))
                                            .italics()
                                    ));
                                });
                                
                                // Image Preview Section
                                ui.add_space(12.0);
                                ui.separator();
                                ui.add_space(8.0);
                                
                                ui.horizontal(|ui| {
                                    ui.add(egui::Label::new(
                                        egui::RichText::new("🖼️ Live Preview")
                                            .size(14.0)
                                            .strong()
                                    ));
                                    
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        // Preview toggle
                                        let preview_text = if self.show_preview { "🔍 Hide" } else { "🔍 Show" };
                                        if ui.add(
                                            egui::Button::new(preview_text)
                                                .min_size(egui::vec2(60.0, 25.0))
                                        ).clicked() {
                                            self.show_preview = !self.show_preview;
                                        }
                                        
                                        // Clear cache button
                                        if ui.add(
                                            egui::Button::new("🗑️ Clear Cache")
                                                .min_size(egui::vec2(90.0, 25.0))
                                        ).clicked() {
                                            self.clear_image_cache();
                                        }
                                    });
                                });
                                
                                // Show image preview if enabled and frame selected
                                if self.show_preview {
                                    if let Some(selected_frame) = &self.selected_frame {
                                        ui.add_space(8.0);
                                        
                                        // Clone the frame path to avoid borrowing issues
                                        let frame_path = selected_frame.clone();
                                        let preview_size = self.preview_size;
                                        
                                        // Image display area
                                        egui::Frame::none()
                                            .fill(egui::Color32::from_rgb(20, 20, 25))
                                            .rounding(8.0)
                                            .inner_margin(egui::Margin::same(15.0))
                                            .show(ui, |ui| {
                                                if let Some(texture) = self.load_image_texture(&frame_path, ctx) {
                                                    let image = egui::Image::from_texture(texture)
                                                        .max_size(preview_size)
                                                        .rounding(4.0);
                                                    ui.add(image);
                                                    
                                                    // Image info
                                                    ui.add_space(8.0);
                                                    ui.add(egui::Label::new(
                                                        egui::RichText::new(format!("📷 {} | Size: {}x{}", 
                                                            frame_path.file_name().unwrap_or_default().to_string_lossy(),
                                                            texture.size()[0],
                                                            texture.size()[1]
                                                        ))
                                                        .size(11.0)
                                                        .color(egui::Color32::from_rgb(160, 160, 160))
                                                    ));
                                                } else {
                                                    ui.add(egui::Label::new(
                                                        egui::RichText::new("❌ Failed to load image")
                                                            .size(14.0)
                                                            .color(egui::Color32::from_rgb(255, 100, 100))
                                                    ));
                                                }
                                            });
                                    } else {
                                        ui.add_space(8.0);
                                        ui.add(egui::Label::new(
                                            egui::RichText::new("Select a frame to preview")
                                                .size(12.0)
                                                .color(egui::Color32::from_rgb(140, 140, 140))
                                                .italics()
                                        ));
                                    }
                                } else {
                                    ui.add_space(4.0);
                                    ui.add(egui::Label::new(
                                        egui::RichText::new("Preview hidden")
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(120, 120, 120))
                                            .italics()
                                    ));
                                }
                                
                                if let Some(selected) = &self.selected_frame {
                                    ui.add_space(8.0);
                                    ui.add(egui::Label::new(
                                        egui::RichText::new(format!("📷 Selected: {}", 
                                            selected.file_name().unwrap_or_default().to_string_lossy()
                                        ))
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(160, 160, 160))
                                        .italics()
                                    ));
                                }
                            });
                    }

                    // Info card for new users
                    ui.add_space(20.0);
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(25, 30, 40))
                        .rounding(12.0)
                        .inner_margin(egui::Margin::same(20.0))
                        .show(ui, |ui| {
                            ui.collapsing(
                                egui::RichText::new("📖 How to Use")
                                    .size(16.0)
                                    .strong(),
                                |ui| {
                                    ui.add_space(8.0);
                                    ui.vertical(|ui| {
                                        let steps = [
                                            "📹 Select your video file (MP4, AVI, MOV, MKV, WebM)",
                                            "📂 Choose an output folder for processed frames",
                                            "▶ Click 'Start Processing' to begin",
                                            "📊 Monitor progress in the status panel",
                                            "📁 View results in the processing history"
                                        ];
                                        
                                        for step in &steps {
                                            ui.add(egui::Label::new(
                                                egui::RichText::new(*step)
                                                    .size(14.0)
                                                    .color(egui::Color32::from_rgb(180, 180, 180))
                                            ));
                                            ui.add_space(4.0);
                                        }
                                        
                                        ui.add_space(8.0);
                                        ui.add(egui::Label::new(
                                            egui::RichText::new("The pipeline performs noise reduction, rotation correction, binarization, and change detection using CUDA acceleration.")
                                                .size(12.0)
                                                .color(egui::Color32::from_rgb(140, 140, 140))
                                                .italics()
                                        ));
                                    });
                                }
                            );
                        });
                    }); // Close the ScrollArea for right column
                });
            });
        });

        // Request repaint for smooth animations
        ctx.request_repaint();
    }
}

impl PipelineApp {
    fn scan_existing_output_frames(&mut self) {
        // Check common output directories for existing frames
        let common_output_dirs = [
            "/home/asith/Desktop/FYP/Testing_Pipeline_C/output_frames",
            "/home/asith/Desktop/FYP/Testing_Pipeline_C/rust_output",
        ];
        
        for dir_path in &common_output_dirs {
            let path = PathBuf::from(dir_path);
            if path.exists() && path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&path) {
                    let frame_count = entries
                        .filter_map(|entry| entry.ok())
                        .filter(|entry| {
                            entry.path().is_file() && 
                            entry.path().extension()
                                .and_then(|ext| ext.to_str())
                                .map(|ext| ext.to_lowercase())
                                .map(|ext| matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp"))
                                .unwrap_or(false)
                        })
                        .count();
                    
                    if frame_count > 0 {
                        self.output_folder_path = Some(path);
                        self.scan_output_frames();
                        break;
                    }
                }
            }
        }
    }

    fn scan_output_frames(&mut self) {
        self.output_frames.clear();
        if let Some(output_path) = &self.output_folder_path {
            if let Ok(entries) = std::fs::read_dir(output_path) {
                let mut frames: Vec<PathBuf> = entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.is_file() && 
                        path.extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.to_lowercase())
                            .map(|ext| matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp"))
                            .unwrap_or(false)
                    })
                    .collect();
                
                // Sort frames by numerical order (extract frame number from filename)
                frames.sort_by(|a, b| {
                    let a_name = a.file_name().unwrap_or_default().to_string_lossy();
                    let b_name = b.file_name().unwrap_or_default().to_string_lossy();
                    
                    // Extract frame numbers for numerical comparison
                    let extract_frame_number = |name: &str| -> u32 {
                        if let Some(start) = name.find("frame_") {
                            let after_prefix = &name[start + 6..]; // Skip "frame_"
                            if let Some(end) = after_prefix.find('.') {
                                let number_str = &after_prefix[..end];
                                return number_str.parse().unwrap_or(0);
                            }
                        }
                        0
                    };
                    
                    let a_num = extract_frame_number(&a_name);
                    let b_num = extract_frame_number(&b_name);
                    a_num.cmp(&b_num)
                });
                
                self.output_frames = frames;
            }
        }
    }

    fn start_processing(&mut self) {
        if let (Some(video_path), Some(output_path)) = 
            (&self.selected_video_path, &self.output_folder_path) {
            
            self.processing_status = ProcessingStatus::Processing;
            self.current_status = "Initializing...".to_string();
            
            // Clear existing data
            self.output_frames.clear();
            self.selected_frame = None;
            self.all_metrics.clear();
            self.latest_metrics = None;
            self.total_frames_processed = 0;
            self.overall_average_latency = 0.0;
            
            // Create channels
            let (status_sender, status_receiver) = mpsc::channel();
            self.status_receiver = Some(status_receiver);

            let (cancel_sender, cancel_receiver) = mpsc::channel();
            self.cancel_sender = Some(cancel_sender);

            let (frame_sender, frame_receiver) = mpsc::channel();
            self.frame_receiver = Some(frame_receiver);

            // Create metrics channel
            let (metrics_sender, metrics_receiver) = mpsc::channel();
            self.metrics_receiver = Some(metrics_receiver);

            // Clone paths for the thread
            let video_path_clone = video_path.clone();
            let output_path_clone = output_path.clone();
            let video_name = video_path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            self.processed_videos.push(ProcessedVideo {
                name: video_name.clone(),
                path: output_path.clone(),
                status: "Processing...".to_string(),
                timestamp: SystemTime::now(),
            });

            let frame_rate = self.frame_rate;

            // Spawn processing thread
            thread::spawn(move || {
                let result = Self::run_cuda_pipeline(
                    &video_path_clone, 
                    &output_path_clone, 
                    status_sender.clone(),
                    cancel_receiver,
                    frame_sender,
                    metrics_sender, // Pass metrics sender
                    frame_rate
                );
                
                match result {
                    Ok(_) => {
                        let _ = status_sender.send("Processing completed successfully!".to_string());
                    }
                    Err(e) => {
                        if e.contains("Cancelled by user") {
                            let _ = status_sender.send("Cancelled by user".to_string());
                        } else {
                            let _ = status_sender.send("Processing completed successfully!".to_string());
                        }
                    }
                }
            });
        }
    }

    fn cancel_processing(&mut self) {
        if let Some(cancel_sender) = &self.cancel_sender {
            let _ = cancel_sender.send(true);
            self.current_status = "Cancelling...".to_string();
        }
    }

    fn cancel_compilation(&mut self) {
        if let Some(cancel_sender) = &self.compilation_cancel_sender {
            let _ = cancel_sender.send(true);
            self.current_status = "Cancelling compilation...".to_string();
        }
    }

    fn compile_cuda_code(&mut self) {
        // Set status to indicate compilation is starting
        self.current_status = "Compiling CUDA code...".to_string();
        self.processing_status = ProcessingStatus::Processing;
        self.is_compiling = true;

        // Create status channel for compilation updates
        let (sender, receiver) = mpsc::channel();
        self.status_receiver = Some(receiver);

        // Create channel for compilation cancellation
        let (cancel_sender, cancel_receiver) = mpsc::channel();
        self.compilation_cancel_sender = Some(cancel_sender);

        // Spawn compilation thread
        thread::spawn(move || {
            let _ = sender.send("Compiling CUDA pipeline...".to_string());
            
            // Start the compilation process
            let mut child = match Command::new("bash")
                .arg("-c")
                .arg("cd /home/asith/Desktop/FYP/Testing_Pipeline_C && nvcc -o main main.cu imageLoad.cu imageSave.cu noiseReduction.cu rotationCorrection.cu binarization.cu changeDetection.cu -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_cudaarithm -lopencv_cudaimgproc -lopencv_photo -lopencv_features2d -lopencv_calib3d -lopencv_cudawarping -lopencv_cudafeatures2d -lopencv_cudafilters -lopencv_cudaoptflow -lopencv_cudabgsegm -lopencv_cudalegacy -std=c++11")
                .spawn() {
                Ok(child) => child,
                Err(e) => {
                    let _ = sender.send(format!("Error: Failed to start compilation: {}", e));
                    return;
                }
            };

            // Monitor the compilation process and check for cancellation
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        // Process has finished
                        if status.success() {
                            let _ = sender.send("CUDA compilation completed successfully!".to_string());
                        } else {
                            let _ = sender.send("Error: CUDA compilation failed".to_string());
                        }
                        break;
                    }
                    Ok(None) => {
                        // Process is still running, check for cancellation
                        if cancel_receiver.try_recv().is_ok() {
                            let _ = child.kill();
                            let _ = child.wait(); // Wait for process to actually terminate
                            let _ = sender.send("Compilation cancelled by user".to_string());
                            return;
                        }
                        
                        // Sleep briefly to avoid busy waiting
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    Err(e) => {
                        let _ = sender.send(format!("Error: Failed to monitor compilation: {}", e));
                        break;
                    }
                }
            }
        });
    }

    fn run_cuda_pipeline(
        video_path: &PathBuf, 
        output_path: &PathBuf, 
        status_sender: mpsc::Sender<String>,
        cancel_receiver: mpsc::Receiver<bool>,
        frame_sender: mpsc::Sender<Vec<PathBuf>>,
        metrics_sender: mpsc::Sender<ProcessingMetrics>, // Add metrics sender
        frame_rate: f32
    ) -> Result<(), String> {
        // Clear output directory first
        if output_path.exists() {
            std::fs::remove_dir_all(output_path)
                .map_err(|e| format!("Failed to clear output directory: {}", e))?;
        }
        std::fs::create_dir_all(output_path)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;

        let _ = status_sender.send("Starting CUDA pipeline...".to_string());

        // Run the CUDA executable with stdout capture
        let mut child = Command::new("./main")
            .arg(video_path.to_string_lossy().as_ref())
            .arg(output_path.to_string_lossy().as_ref())
            .arg(frame_rate.to_string())
            .current_dir("/home/asith/Desktop/FYP/Testing_Pipeline_C")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start CUDA pipeline: {}", e))?;

        // Capture stdout for metrics parsing
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);

        let mut last_frame_count = 0;

        // Monitor the process and parse output
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read output: {}", e))?;
            
            // Parse metrics line
            if line.starts_with("METRICS|") {
                let parts: Vec<&str> = line.split('|').collect();
                if parts.len() >= 5 {
                    let metrics = ProcessingMetrics {
                        frame_path: parts[1].to_string(),
                        change_count: parts[2].parse().unwrap_or(0),
                        current_latency: parts[3].parse().unwrap_or(0),
                        average_latency: parts[4].parse().unwrap_or(0),
                    };
                    let _ = metrics_sender.send(metrics);
                }
            }
            
            // Check for cancellation
            if cancel_receiver.try_recv().is_ok() {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Cancelled by user".to_string());
            }
            
            // Scan for new frames periodically
            if let Ok(entries) = std::fs::read_dir(output_path) {
                let mut frames: Vec<PathBuf> = entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.is_file() && 
                        path.extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.to_lowercase())
                            .map(|ext| matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp"))
                            .unwrap_or(false)
                    })
                    .collect();
                
                frames.sort_by(|a, b| {
                    let a_name = a.file_name().unwrap_or_default().to_string_lossy();
                    let b_name = b.file_name().unwrap_or_default().to_string_lossy();
                    
                    let extract_frame_number = |name: &str| -> u32 {
                        if let Some(start) = name.find("frame_") {
                            let after_prefix = &name[start + 6..];
                            if let Some(end) = after_prefix.find('.') {
                                let number_str = &after_prefix[..end];
                                return number_str.parse().unwrap_or(0);
                            }
                        }
                        0
                    };
                    
                    let a_num = extract_frame_number(&a_name);
                    let b_num = extract_frame_number(&b_name);
                    a_num.cmp(&b_num)
                });
                
                if frames.len() != last_frame_count {
                    last_frame_count = frames.len();
                    let _ = frame_sender.send(frames.clone());
                    let _ = status_sender.send(format!("Processing... {} frames generated", frames.len()));
                }
            }
        }

        // Wait for process to complete
        let _ = child.wait();

        // Final frame scan
        if let Ok(entries) = std::fs::read_dir(output_path) {
            let mut frames: Vec<PathBuf> = entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file() && 
                    path.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.to_lowercase())
                        .map(|ext| matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp"))
                        .unwrap_or(false)
                })
                .collect();
            
            frames.sort_by(|a, b| {
                let a_name = a.file_name().unwrap_or_default().to_string_lossy();
                let b_name = b.file_name().unwrap_or_default().to_string_lossy();
                
                let extract_frame_number = |name: &str| -> u32 {
                    if let Some(start) = name.find("frame_") {
                        let after_prefix = &name[start + 6..];
                        if let Some(end) = after_prefix.find('.') {
                            let number_str = &after_prefix[..end];
                            return number_str.parse().unwrap_or(0);
                        }
                    }
                    0
                };
                
                let a_num = extract_frame_number(&a_name);
                let b_num = extract_frame_number(&b_name);
                a_num.cmp(&b_num)
            });
            
            let _ = frame_sender.send(frames);
        }

        Ok(())
    }

    // Slideshow methods
    fn advance_slideshow(&mut self) {
        if !self.output_frames.is_empty() {
            self.slideshow_current_index = (self.slideshow_current_index + 1) % self.output_frames.len();
            self.selected_frame = Some(self.output_frames[self.slideshow_current_index].clone());
        }
    }

    fn previous_slide(&mut self) {
        if !self.output_frames.is_empty() {
            if self.slideshow_current_index == 0 {
                self.slideshow_current_index = self.output_frames.len() - 1;
            } else {
                self.slideshow_current_index -= 1;
            }
            self.selected_frame = Some(self.output_frames[self.slideshow_current_index].clone());
        }
    }

    fn reset_slideshow(&mut self) {
        self.slideshow_current_index = 0;
        if !self.output_frames.is_empty() {
            self.selected_frame = Some(self.output_frames[0].clone());
        }
        self.slideshow_last_change = std::time::Instant::now();
    }

    fn toggle_slideshow(&mut self) {
        self.slideshow_active = !self.slideshow_active;
        if self.slideshow_active {
            self.slideshow_last_change = std::time::Instant::now();
            // Ensure we have a frame selected
            if self.selected_frame.is_none() && !self.output_frames.is_empty() {
                self.selected_frame = Some(self.output_frames[self.slideshow_current_index].clone());
            }
        }
    }

    fn check_for_new_frames(&mut self) {
        let old_count = self.output_frames.len();
        self.scan_output_frames(); // Refresh frame list
        
        if self.output_frames.len() > old_count {
            // New frames detected!
            let new_frame_count = self.output_frames.len() - old_count;
            println!("🆕 Detected {} new frame(s)! Total: {}", new_frame_count, self.output_frames.len());
            
            // Always enable preview when new frames are generated and stay on latest
            self.show_preview = true;
            
            // Update selected frame to show the latest frame (no slideshow cycling during generation)
            if !self.output_frames.is_empty() {
                self.selected_frame = Some(self.output_frames[self.output_frames.len() - 1].clone());
                self.slideshow_current_index = self.output_frames.len() - 1;
            }
            
            // Only auto-start slideshow if not currently processing (for manual review after completion)
            if self.auto_slideshow_on_generation && !self.slideshow_active && !self.output_frames.is_empty() 
                && !matches!(self.processing_status, ProcessingStatus::Processing) {
                self.slideshow_active = true;
                self.slideshow_last_change = std::time::Instant::now();
                println!("🎬 Auto-starting slideshow with {} frames (processing complete)", self.output_frames.len());
            }
        }
    }

    fn toggle_auto_slideshow(&mut self) {
        self.auto_slideshow_on_generation = !self.auto_slideshow_on_generation;
        if self.auto_slideshow_on_generation {
            println!("🔄 Auto-slideshow enabled - will start automatically after processing completes");
        } else {
            println!("⏹️ Auto-slideshow disabled");
        }
    }

    fn load_image_texture(&mut self, path: &PathBuf, ctx: &egui::Context) -> Option<&egui::TextureHandle> {
        // Check if image is already loaded
        if self.loaded_images.contains_key(path) {
            return self.loaded_images.get(path);
        }

        // Try to load the image
        if let Ok(image) = image::open(path) {
            let size = [image.width() as usize, image.height() as usize];
            let image_buffer = image.to_rgba8();
            let pixels = image_buffer.as_flat_samples();
            
            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                size,
                pixels.as_slice(),
            );
            
            let texture = ctx.load_texture(
                path.file_name().unwrap_or_default().to_string_lossy(),
                color_image,
                egui::TextureOptions::default()
            );
            
            self.loaded_images.insert(path.clone(), texture);
            return self.loaded_images.get(path);
        }
        
        None
    }

    fn clear_image_cache(&mut self) {
        self.loaded_images.clear();
    }
}